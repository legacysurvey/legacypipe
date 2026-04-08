import queue
import threading
import multiprocessing

from multiprocessing import get_context, util
from multiprocessing.pool import Pool, _PoolCache, mapstar, INIT, RUN, TERMINATE
from legacypipe.trackingpool import (TrackingPool, MyList, TrackingIMapUnorderedIterator,
                                     worker)

# Two kinds of priority pools one could want:

# - blobs being processed by GPU and CPU workers; one
#   imap_unordered(), with priorities.  GPU pulls from the
#   high-priority end, CPU pulls from the low-priority end.

# --> I think you can achieve this by having two pools (GPU, CPU),
# with two iterators that are chomping work off of either end of a
# (shared) priority list.  This does not require any changes to the
# Pool structure.

# - blob-parallel mode, where many blobs are being processed, and
#   within that, non-overlapping sources are being processed in
#   parallel in one big pool that is shared between the blobs.  You
#   then have many imap_unordered() calls to fit_one() and
#   model_select_one(); you want to pool them all together and have
#   the GPU workers pull the high-priority calls, and the CPU workers
#   pull the low-priority calls (or, perhaps, the oldest, or some
#   combination of priority terms).

# --> In this case, you need to modify the Pool, because work items
# from multiple imap_unordered() function calls are going into one
# giant priority queue.  You then need the Pool to know about the two
# different kinds of workers, each with its own initialization
# functions/args, and number of workers.

# Does the _handle_tasks function need to change?  In the normal Pool
# setup, the _handle_tasks routine calls the input iterator only as
# needed, one element at a time.  Well, actually, it pulls them in a
# loop and puts them onto the "self._inqueue" via the
# "self._quick_put" -> "put" method.  That _inqueue is a
# SimpleQueue().  SimpleQueue uses a connection.Pipe(duplex=False) ->
# an os.pipe.

# In pool.py, the map() and several other calls run list(iterable) on
# the input before sending it to the task queue, thus fetching all
# work before starting.  Imap_unordered, on the other hand, puts the
# iterator onto the task queue, and then the task queue iterates work
# as required, and put()s work until the _inqueue is full.  This
# buffer appears to be tens of kB in size.  The put() call is what
# blocks.

# For a shared priority queue, I think we could either have two
# _inqueues, one for high-priority work and a second for low-priority
# work, and use a poll() or something like that to put() work onto
# both of them.  Or maybe two _handle_tasks threads, pulling for
# opposite ends of the priority queue and put()ting onto their
# respective pipes.

# BUT, that means that we are buffering (an unknown number of) tasks
# in the _inqueue pipes.  If a higher-priority task comes in, it
# cannot bump one in the pipe.  When we receive a new imap() call, we
# would also have to be careful to insert all the work at once, to
# avoid races where as soon as we enqueue one item, the high-priority
# and low-priority _handle_tasks threads race to grab it and put it
# onto their _inqueues.

# MAYBE we want to actually just have a Manager type object that the
# workers can pull work from.  That way, there is no buffer of tasks,
# workers always get the best task available at the moment.  This does
# make it synchronous, though.

class PriorityPool(TrackingPool):

    def __init__(self, n_high_priority, n_low_priority,
                 initialize_high_priority=None,
                 initialize_low_priority=None,
                 initialize_high_priority_args=None,
                 initialize_low_priority_args=None,
                 maxtasksperchild=None, context=None,
                 raise_deadworker_exception=True):
        # The priority queue is accessible to all the worker
        # processes, so must be a proxy object.
        self._pq_manager = PQManager()
        self._pq_manager.start()
        self._pq = self._pq_manager.threadsafe_priority_deque()

        self._pool = MyList()
        self._pool.next_worker_id = 1000
        self._worker_pids = {}
        self._state = INIT
        self._raise_deadworker_exception = raise_deadworker_exception

        self._ctx = context or get_context()
        self._setup_queues()

        # We just set the _inqueue to this one, because it already
        # gets passed to all the places it needs to go!
        self._inqueue = self._pq
        # When the task_handler finishes, it sends a bunch of Nones into the _inqueue
        # to tell the workers to stop.  We give them a fake priority.
        def quick_put(x):
            if x is None:
                self._pq.insert_one(0, x)
        self._quick_put = quick_put

        # We actually don't use the taskqueue!!  Because we're instead sharing _inqueue
        # directly with the workers.  We *could* have the task_handler manage inserting
        # tasks into the _inqueue, more similar to how the regular Pools work.
        self._taskqueue = queue.SimpleQueue()

        # The _change_notifier queue exist to wake up self._handle_workers()
        # when the cache (self._cache) is empty or when there is a change in
        # the _state variable of the thread that runs _handle_workers.
        self._change_notifier = self._ctx.SimpleQueue()
        self._cache = _PoolCache(notifier=self._change_notifier)
        self._maxtasksperchild = maxtasksperchild
        self._initialize_hi = initialize_high_priority
        self._initargs_hi = initialize_high_priority_args
        self._initialize_lo = initialize_low_priority
        self._initargs_lo = initialize_low_priority_args

        if (n_high_priority + n_low_priority) < 1:
            raise ValueError("Number of processes must be at least 1")
        if maxtasksperchild is not None:
            if not isinstance(maxtasksperchild, int) or maxtasksperchild <= 0:
                raise ValueError("maxtasksperchild must be a positive int or None")

        if initialize_high_priority is not None and not callable(initialize_high_priority):
            raise TypeError('initialize_high_priority must be a callable')
        if initialize_low_priority is not None and not callable(initialize_low_priority):
            raise TypeError('initialize_low_priority must be a callable')

        self._n_hi = n_high_priority
        self._n_lo = n_low_priority
        try:
            self._repopulate_pool()
        except Exception:
            for w in self._pool:
                if w.exitcode is None:
                    w.terminate()
            for w in self._pool:
                w.join()
            raise

        sentinels = self._get_sentinels()

        self._worker_handler = threading.Thread(
            target=PriorityPool._my_handle_workers,
            args=(self._cache, self._taskqueue, self._ctx, self.Process,
                  self._n_hi, self._n_lo, self._pool, self._inqueue, self._outqueue,
                  self._initialize_hi, self._initargs_hi,
                  self._initialize_lo, self._initargs_lo,
                  self._maxtasksperchild,
                  self._wrap_exception, sentinels, self._change_notifier,
                  self._worker_pids)
            )
        self._worker_handler.daemon = True
        self._worker_handler._state = RUN
        self._worker_handler.start()

        self._task_handler = threading.Thread(
            target=Pool._handle_tasks,
            args=(self._taskqueue, self._quick_put, self._outqueue,
                  self._pool, self._cache)
            )
        self._task_handler.daemon = True
        self._task_handler._state = RUN
        self._task_handler.start()

        self._result_handler = threading.Thread(
            target=TrackingPool._my_handle_results,
            args=(self._outqueue, self._quick_get, self._cache, self._worker_pids,
                  self._raise_deadworker_exception)
            )
        self._result_handler.daemon = True
        self._result_handler._state = RUN
        self._result_handler.start()

        self._terminate = util.Finalize(
            self, self._terminate_pool,
            args=(self._taskqueue, self._inqueue, self._outqueue, self._pool,
                  self._change_notifier, self._worker_handler, self._task_handler,
                  self._result_handler, self._cache),
            exitpriority=15
            )
        self._state = RUN

    def terminate(self):
        super().terminate()
        if self._pq_manager is not None:
            self._pq_manager.shutdown()
            self._pq_manager = None

    def _setup_queues(self):
        # no _inqueue or _quick_put
        self._outqueue = self._ctx.SimpleQueue()
        self._quick_get = self._outqueue._reader.recv

    @staticmethod
    def _help_stuff_finish(inqueue, task_handler, size):
        # in pool.py this clear out the _inqueue to prevent task_handler being locked;
        # we don't need that.
        pass

    def _repopulate_pool(self):
        self._repopulate_pool_static(self._ctx, self.Process,
                                     self._n_hi, self._n_lo,
                                     self._pool, self._inqueue, self._outqueue,
                                     self._initialize_hi, self._initargs_hi,
                                     self._initialize_lo, self._initargs_lo,
                                     self._maxtasksperchild,
                                     self._wrap_exception,
                                     self._worker_pids)
    @staticmethod
    def _repopulate_pool_static(ctx, Process, n_hi, n_lo, pool, inqueue, outqueue,
                                initialize_hi, initargs_hi, initialize_lo, initargs_lo,
                                maxtasksperchild, wrap_exception, pids):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for n, high, init, initargs in [(n_hi, True,  initialize_hi, initargs_hi),
                                        (n_lo, False, initialize_lo, initargs_lo),]:
            n_create = n - len([p for p in pool if p.high_priority == high])
            pstr = ('high' if high else 'low')
            util.debug('Creating %i %s priority workers' % (n_create, pstr))
            for _ in range(n_create):
                worker_id = pool.next_worker_id
                pool.next_worker_id += 1

                fake_inqueue = FakeInqueue(inqueue, high)
                w = Process(ctx, target=worker,
                            args=(fake_inqueue, outqueue,
                                  init, initargs, maxtasksperchild,
                                  wrap_exception,
                                  worker_id))
                w.name = w.name.replace('Process', 'PoolWorker(%s)' % pstr)
                w.daemon = True
                w.worker_id = worker_id
                w.high_priority = high
                w.start()
                pool.append(w)
                pid = w.pid
                pids[worker_id] = pid
                util.debug('added worker %i with pid %s' % (worker_id, pid))

    @staticmethod
    def _my_maintain_pool(ctx, Process, n_hi, n_lo, pool, inqueue, outqueue,
                          initializer_hi, initargs_hi,
                          initializer_lo, initargs_lo,
                          maxtasksperchild,
                          wrap_exception, pids):
        """Clean up any exited workers and start replacements for them.
        """
        cleaned = TrackingPool._join_exited_workers(pool)
        if cleaned:
            for worker_id, exitcode, _ in cleaned:
                # notify the handle_results() thread that this worker has died;
                # whatever it was working on is toast
                outqueue.put((None, None, False, worker_id, exitcode))
            PriorityPool._repopulate_pool_static(ctx, Process, n_hi, n_lo, pool,
                                                 inqueue, outqueue,
                                                 initializer_hi, initargs_hi,
                                                 initializer_lo, initargs_lo,
                                                 maxtasksperchild,
                                                 wrap_exception, pids)
    @classmethod
    def _my_handle_workers(cls, cache, taskqueue, ctx, Process, n_hi, n_lo,
                           pool, inqueue, outqueue,
                           initializer_hi, initargs_hi,
                           initializer_lo, initargs_lo,
                           maxtasksperchild, wrap_exception, sentinels,
                           change_notifier, pids):
        thread = threading.current_thread()

        # Keep maintaining workers until the cache gets drained, unless the pool
        # is terminated.
        while thread._state == RUN or (cache and thread._state != TERMINATE):
            cls._my_maintain_pool(ctx, Process, n_hi, n_lo, pool,
                                  inqueue, outqueue,
                                  initializer_hi, initargs_hi,
                                  initializer_lo, initargs_lo,
                                  maxtasksperchild, wrap_exception, pids)
            current_sentinels = [*cls._get_worker_sentinels(pool), *sentinels]
            cls._wait_for_updates(current_sentinels, change_notifier)
        # send sentinel to stop workers
        taskqueue.put(None)
        util.debug('worker handler exiting')

    def priority_imap_unordered(self, func, iterable):
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)
        result = TrackingIMapUnorderedIterator(self)
        job_num = result._job
        result._set_length(len(iterable))
        self._pq.insert_many([(p, (job_num, i, func, (arg,), {}))
                              for i,(p,arg) in enumerate(iterable)])
        return result

    def imap_unordered(self, func, iterable):
        self.priority_imap_unordered(func, [(0,x) for x in iterable])

    def apply(self, *args, **kwargs):
        raise RuntimeError('prioritypool.apply()')
    def map(self, *args, **kwargs):
        raise RuntimeError('prioritypool.map()')
    def starmap(self, *args, **kwargs):
        raise RuntimeError('prioritypool.starmap()')
    def starmap_async(self, *args, **kwargs):
        raise RuntimeError('prioritypool.starmap_async()')
    def imap(self, *args, **kwargs):
        raise RuntimeError('prioritypool.imap()')
    def apply_async(self, *args, **kwargs):
        raise RuntimeError('prioritypool.apply_async()')
    def map_async(self, *args, **kwargs):
        raise RuntimeError('prioritypool.map_async()')

# make it so we can reuse the worker() function from TrackingPool.  It gets work by
# calling the _inqueue.get() method.  So fake that.
class FakeInqueue(object):
    def __init__(self, pq, high):
        self.pq = pq
        self.high = high
    def get(self):
        if self.high:
            _,val = self.pq.get_high_priority()
            return val
        _,val = self.pq.get_low_priority()
        return val

from collections import deque
import numpy as np

# A very basic priority double-ended queue
class priority_deque(deque):
    def _binary_search(self, prio):
        # binary search for insertion index
        lo,hi = 0, len(self)
        while True:
            if lo == hi:
                return lo
            mid = (lo + hi) // 2
            mid_prio = self[mid][0]
            if mid_prio < prio:
                lo = mid+1
            else:
                hi = mid

    def insert_sorted(self, prio, val):
        i = self._binary_search(prio)
        self.insert(i, (prio, val))

# thread-safe wrapper on the priority queue that blocks on inserts and gets
class threadsafe_priority_deque(priority_deque):
    def __init__(self):
        super().__init__()
        self._cond = threading.Condition()
    def insert_many(self, vals):
        with self._cond:
            # If we wanted a maximum size, could check that here,
            # and cond.wait() for space to be available.
            for p,v in vals:
                self.insert_sorted(p, v)
            self._cond.notify_all()
    def insert_one(self, p, val):
        with self._cond:
            self.insert_sorted(p, val)
            self._cond.notify()
    def get_low_priority(self):
        with self._cond:
            while len(self) == 0:
                self._cond.wait()
            return self.popleft()
    def get_high_priority(self):
        with self._cond:
            while len(self) == 0:
                self._cond.wait()
            return self.pop()

from multiprocessing.managers import BaseManager
class PQManager(BaseManager):
    pass
PQManager.register('threadsafe_priority_deque', threadsafe_priority_deque)

#################################
#   Testing
#################################

class MyArgs(object):
    def __init__(self, val):
        self.val = val
    def __str__(self):
        return 'MyArgs: ' + self.val[0]
    def get(self):
        return self.val

def priority_work_generator(n):
    for _ in range(n):
        p = np.random.randint(0, 1000)
        yield p, MyArgs(np.zeros(10, int) + p)

is_high_priority_worker = None

def init_hi(args):
    print('init_hi:', args)
    global is_high_priority_worker
    is_high_priority_worker = True
def init_lo(args):
    print('init_lo:', args)
    global is_high_priority_worker
    is_high_priority_worker = False

def func(i):
    import time
    print('func starting (%s prio):' % ('high' if is_high_priority_worker else 'low'),
          i.get()[0])
    time.sleep(1)
    return i.get()[0]

def func2(i):
    import time
    print('func2 starting (%s prio):' % ('high' if is_high_priority_worker else 'low'),
          i.get()[0])
    time.sleep(1)
    return i.get()[0]

if __name__ == '__main__':
    import logging
    import sys
    import multiprocessing.util as mpu
    #logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stdout)
    #mpu.log_to_stderr(level=logging.DEBUG)

    p = PriorityPool(2, 2,
                     initialize_high_priority=init_hi,
                     initialize_low_priority=init_lo,
                     initialize_high_priority_args=(42,),
                     initialize_low_priority_args=(37,)
                     )

    r1 = p.priority_imap_unordered(func, priority_work_generator(10))
    r2 = p.priority_imap_unordered(func2, priority_work_generator(10))
    print('r1', r1)
    print('r2', r2)
    for ri in r1:
        print('result 1', ri)
    for ri in r2:
        print('result 2', ri)
    print('Got all results')
    print('close')
    p.close()
    import time
    time.sleep(1.)
    print('terminate')
    p.terminate()
    print('join')
    p.join()

    sys.exit(0)

    if False:
        q = priority_deque()

        q.append((0,0))
        q.append((2,0))
        #q.insert_sorted(-1,0)
        q.insert_sorted(-1,77)
        print(q)
        import sys
        sys.exit(0)

        for i in range(100):
            prio = np.random.randint(0, 4)
            q.insert_sorted(prio, i)
            print(q)

