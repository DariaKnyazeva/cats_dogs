import asyncio
import unittest


class AsyncTestCase(unittest.TestCase):
    """
    Base class for tests that checks asynchronous codes
    """

    def get_async_result(self, func, *func_args, **func_kwargs):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(func(*func_args, **func_kwargs))
        loop.close()

        return result
