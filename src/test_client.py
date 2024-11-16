import asyncio
import websockets

async def listen():
    url = "ws://127.0.0.1:8081"
    try:
        # Connect to the server
        async with websockets.connect(url) as ws:
            # Send a greeting message
            await ws.send("Hello Server!")
            i = 1
            while True:
                try:
                    msg = await ws.recv()
                    print(msg)
                    if i <= 100:
                        i += 1
                    else:
                        print("Stop Test Client")
                        await ws.close(1000, "Close it now")
                        break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed: {e}")
                    break
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    try:
        # asyncio.get_event_loop().run_until_complete(listen()) 대신
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("Client stopped by user")
    except Exception as e:
        print(f"Error: {e}")