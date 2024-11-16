import asyncio
import websockets
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def listen():
    url = "ws://mvp_simulation:8081"
    retries = 5
    attempt = 0

    while attempt < retries:
        try:
            # Connect to the server
            async with websockets.connect(url) as ws:
                logger.info("Connected to the server!")
                # Send a greeting message
                await ws.send("Hello Server!")
                while True:
                    try:
                        msg = await ws.recv()
                        logger.info(f"Received message: {msg}")
                        time.sleep(2)
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"Connection closed: {e}")
                        break

                logger.info("Stop Test Client")
                await ws.close(1000, "Close it now")
                break  
        except Exception as e:
            logger.error(f"Connection error: {e}")
            attempt += 1
            if attempt < retries:
                logger.info("Retrying...")
                await asyncio.sleep(2)
            else:
                logger.error("Failed to connect after multiple attempts.")
                break

if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")