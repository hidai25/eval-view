# evalview/adapters/webhook_adapter.py
import asyncio
import aiohttp
from aiohttp import web
from evalview.adapters.base import AgentAdapter
from evalview.execution_trace import ExecutionTrace

class WebhookAdapter(AgentAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.port = config.get('webhook_port', 9999)
        self.timeout = config.get('webhook_timeout', 30)
        self.app = web.Application()
        self.app.router.add_post('/', self.handle_callback)
        self.runner = None
        self.loop = asyncio.get_event_loop()
        self.response_received = asyncio.Event()

    async def start_server(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, 'localhost', self.port)
        await site.start()

    async def stop_server(self):
        await self.runner.cleanup()

    async def handle_callback(self, request):
        payload = await request.json()
        self.execution_trace = ExecutionTrace.from_payload(payload)
        self.response_received.set()
        return web.Response(text="Callback received")

    async def send_query(self, query, context):
        await self.start_server()
        context['callback_url'] = f'http://localhost:{self.port}'
        async with aiohttp.ClientSession() as session:
            async with session.post(self.config['endpoint'], json={'query': query, 'context': context}) as response:
                await response.text()
        await asyncio.wait_for(self.response_received.wait(), self.timeout)
        await self.stop_server()
        return self.execution_trace

    def execute(self, query, context):
        return self.loop.run_until_complete(self.send_query(query, context))