import asyncio
import websockets

async def listen():
    url = "ws://127.0.0.1:8081"
    # Connect to the server
    async with websockets.connect(url) as ws:
        # Send a greeting message
        await ws.send("Hello Server!")

        i = 1
        while True:
            msg = await ws.recv()
            print(msg)
            if i <= 100:
                i += 1
            else:
                print("Stop Test Client")
                await ws.close(1000, "Close it now")
                break



# Start the connection
asyncio.get_event_loop().run_until_complete(listen())