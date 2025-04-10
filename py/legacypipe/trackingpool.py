import os
import time
import queue
import threading
import multiprocessing
from multiprocessing.pool import Pool, IMapUnorderedIterator, _PoolCache, INIT, RUN, TERMINATE
from multiprocessing import get_context, util

class TrackingIMapUnorderedIterator(IMapUnorderedIterator):
    def __init__(self, pool):
        super().__init__(pool)
        self._status = {}
    def _set_status(self, i, s):
        self._status[i] = s
    def _set(self, i, obj):
        if i in self._status:
            del self._status[i]
        super()._set(i, obj)
    def get_running_jobs(self):
        return self._status

def worker(inqueue, outqueue, initializer=None, initargs=(), maxtasks=None,
           wrap_exception=False):
    if (maxtasks is not None) and not (isinstance(maxtasks, int)
                                       and maxtasks >= 1):
        raise AssertionError("Maxtasks {!r} is not valid".format(maxtasks))
    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()

    if initializer is not None:
        initializer(*initargs)

    completed = 0
    while maxtasks is None or (maxtasks and completed < maxtasks):
        try:
            task = get()
        except (EOFError, OSError):
            util.debug('worker got EOFError or OSError -- exiting')
            break

        if task is None:
            util.debug('worker got sentinel -- exiting')
            break

        job, i, func, args, kwds = task
        #
        put((job, i, False, dict(event='start', pid=os.getpid(), time=time.time())))
        #
        try:
            result = (True, func(*args, **kwds))
        except Exception as e:
            if wrap_exception and func is not _helper_reraises_exception:
                e = ExceptionWithTraceback(e, e.__traceback__)
            result = (False, e)
        try:
            #
            #put((job, i, result))
            put((job, i, True, result))
            #
        except Exception as e:
            wrapped = MaybeEncodingError(e, result[1])
            util.debug("Possible encoding error while sending result: %s" % (
                wrapped))
            #
            #put((job, i, (False, wrapped)))
            put((job, i, True, (False, wrapped)))
            #
        task = job = result = func = args = kwds = None
        completed += 1
    util.debug('worker exiting after %d tasks' % completed)

class TrackingPool(Pool):

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None):
        # Attributes initialized early to make sure that they exist in
        # __del__() if __init__() raises an exception
        self._pool = []
        self._state = INIT

        self._ctx = context or get_context()
        self._setup_queues()
        self._taskqueue = queue.SimpleQueue()
        # The _change_notifier queue exist to wake up self._handle_workers()
        # when the cache (self._cache) is empty or when there is a change in
        # the _state variable of the thread that runs _handle_workers.
        self._change_notifier = self._ctx.SimpleQueue()
        self._cache = _PoolCache(notifier=self._change_notifier)
        self._maxtasksperchild = maxtasksperchild
        self._initializer = initializer
        self._initargs = initargs

        if processes is None:
            processes = os.cpu_count() or 1
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")
        if maxtasksperchild is not None:
            if not isinstance(maxtasksperchild, int) or maxtasksperchild <= 0:
                raise ValueError("maxtasksperchild must be a positive int or None")

        if initializer is not None and not callable(initializer):
            raise TypeError('initializer must be a callable')

        self._processes = processes
        try:
            self._repopulate_pool()
        except Exception:
            for p in self._pool:
                if p.exitcode is None:
                    p.terminate()
            for p in self._pool:
                p.join()
            raise

        sentinels = self._get_sentinels()

        self._worker_handler = threading.Thread(
            target=Pool._handle_workers,
            args=(self._cache, self._taskqueue, self._ctx, self.Process,
                  self._processes, self._pool, self._inqueue, self._outqueue,
                  self._initializer, self._initargs, self._maxtasksperchild,
                  self._wrap_exception, sentinels, self._change_notifier)
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
            target=TrackingPool._handle_results,
            args=(self._outqueue, self._quick_get, self._cache)
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

    # From Python 3.12.7 : pool.py
    def imap_unordered(self, func, iterable, chunksize=1):
        '''
        Like `imap()` method but ordering of results is arbitrary.
        '''
        self._check_running()
        if chunksize == 1:
            result = TrackingIMapUnorderedIterator(self)
            #result = IMapUnorderedIterator(self)
            self._taskqueue.put(
                (
                    self._guarded_task_generation(result._job, func, iterable),
                    result._set_length
                ))
            return result
        else:
            if chunksize < 1:
                raise ValueError(
                    "Chunksize must be 1+, not {0!r}".format(chunksize))
            task_batches = Pool._get_tasks(func, iterable, chunksize)
            result = IMapUnorderedIterator(self)
            self._taskqueue.put(
                (
                    self._guarded_task_generation(result._job,
                                                  mapstar,
                                                  task_batches),
                    result._set_length
                ))
            return (item for chunk in result for item in chunk)

    # From Python 3.12.7 : pool.py
    @staticmethod
    def _handle_results(outqueue, get, cache):
        thread = threading.current_thread()

        while 1:
            try:
                task = get()
            except (OSError, EOFError):
                util.debug('result handler got EOFError/OSError -- exiting')
                return

            if thread._state != RUN:
                assert thread._state == TERMINATE, "Thread not in TERMINATE"
                util.debug('result handler found thread._state=TERMINATE')
                break

            if task is None:
                util.debug('result handler got sentinel')
                break

            status_update = False
            if len(task) == 4:
                job, i, done, obj = task
                status_update = not(done)
            else:
                job, i, obj = task

            if status_update:
                try:
                    r = cache[job]
                    if isinstance(r, TrackingIMapUnorderedIterator):
                        r._set_status(i, obj)
                except:
                    import traceback
                    print('_handle_results failed to set status:')
                    traceback.print_exc()
                    pass
            else:
                try:
                    cache[job]._set(i, obj)
                except KeyError:
                    pass
            task = job = obj = None

        while cache and thread._state != TERMINATE:
            try:
                task = get()
            except (OSError, EOFError):
                util.debug('result handler got EOFError/OSError -- exiting')
                return

            if task is None:
                util.debug('result handler ignoring extra sentinel')
                continue
            job, i, obj = task
            try:
                cache[job]._set(i, obj)
            except KeyError:
                pass
            task = job = obj = None

        if hasattr(outqueue, '_reader'):
            util.debug('ensuring that outqueue is not full')
            # If we don't make room available in outqueue then
            # attempts to add the sentinel (None) to outqueue may
            # block.  There is guaranteed to be no more than 2 sentinels.
            try:
                for i in range(10):
                    if not outqueue._reader.poll():
                        break
                    get()
            except (OSError, EOFError):
                pass

        util.debug('result handler exiting: len(cache)=%s, thread._state=%s',
              len(cache), thread._state)
    
    def _repopulate_pool(self):
        return self._repopulate_pool_static(self._ctx, self.Process,
                                            self._processes,
                                            self._pool, self._inqueue,
                                            self._outqueue, self._initializer,
                                            self._initargs,
                                            self._maxtasksperchild,
                                            self._wrap_exception)

    @staticmethod
    def _repopulate_pool_static(ctx, Process, processes, pool, inqueue,
                                outqueue, initializer, initargs,
                                maxtasksperchild, wrap_exception):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(processes - len(pool)):
            w = Process(ctx, target=worker,
                        args=(inqueue, outqueue,
                              initializer,
                              initargs, maxtasksperchild,
                              wrap_exception))
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            pool.append(w)
            util.debug('added worker')



def test_input_generator(n, job_id_map):
    for i in range(n):
        import numpy as np
        x = np.random.random((1000,1000))

        r = 1000 + i
        job_id_map[i] = r
        print('Yielding input', r)
        yield (r,x)

def test_sleep(x):
    import numpy as np
    (i,a) = x
    print('Starting', i, 'in pid', os.getpid())
    import time
    time.sleep(3. + np.random.random()*3)
    print('Done', i)
    return x

if __name__ == '__main__':
    import time

    job_id_map = {}
    in_iter = test_input_generator(100, job_id_map)
    with TrackingPool(4) as pool:
        out_iter = pool.imap_unordered(test_sleep, in_iter)
        while True:
            try:
                try:
                    #r = next(out_iter, 1.)
                    r = out_iter.next(1.)
                except multiprocessing.TimeoutError:
                    s = out_iter._status
                    tnow = time.time()
                    print('Waiting:')
                    for i,st in s.items():
                        pid = st['pid']
                        tstart = st['time']
                        print('  ', job_id_map[i], ': running in PID', pid, 'for %.1f sec' % (tnow - tstart))
                    continue
                i,x = r
                print('Got result', i)
                time.sleep(0.01)
                print('out_iter status:', out_iter._status)
            except StopIteration:
                print('StopIteration')
                break
