import motorcortex
import time
import sys 


# Define the callback function that will be called whenever a message is received
def callback_generator(mjc_simulator):
    def build_q_from_mtx(parameters):
        print("Callback")
        print(parameters)
        q_from_mtx = []
        for cnt in range(0, len(parameters)):
            param = parameters[cnt]
            timestamp = param.timestamp.sec + param.timestamp.nsec * 1e-9
            value = param.value
            # print the timestamp and value; convert the value to a string first
            # so we do not need to check all types before printing it
            q_from_mtx.append(value)

        mjc_simulator.handle_control_from_mtx(q_from_mtx)
        print(f"Callback, timestamp: {timestamp} value: {q_from_mtx}")
    
    return build_q_from_mtx

def callback(msg):
    print(msg)

def Initialize_cnx():
    parameter_tree = motorcortex.ParameterTree()
    # Open request and subscribe connection
    try:
        req, sub = motorcortex.connect("wss://192.168.56.101:5568:5567", 
                                       motorcortex.MessageTypes(), parameter_tree,
                                       certificate="mcx.cert.crt", timeout_ms=1000,
                                       login="admin", password="vectioneer")
    except RuntimeError as err:
        print(err)
        sys.exit()
    
    return req, sub


def main(mujoco_simulator):
    req, sub  = Initialize_cnx()
    paths = ['root/AxesControl/axesPositionsActual']
    # define the frequency divider that tells the server to publish only every
    # n-th sample. This depends on the update rate of the publisher.
    divider = 100
    # subscribe and wait for the reply with a timeout of 10 seconds
    subscription = sub.subscribe(paths, 'group1', divider)
    # get reply from the server
    is_subscribed = subscription.get()
    # print subscription status and layout
    if (is_subscribed is not None) and (is_subscribed.status == motorcortex.OK):
      print(f"Subscription successful, layout: {subscription.layout()}")
    else:
      print(f"Subscription failed, do your paths exist? \npaths: {paths}")
      sub.close()
      exit()
      
    # set the callback function that handles the received data
    # Note that this is a non-blocking call, starting a new thread that handles
    # the messages. You should keep the application alive for a s long as you need to
    # receive the messages

    callback_function = callback_generator(mujoco_simulator)
    subscription.notify(callback_function) #build the callback
    #subscription.notify(callback)
    # polling subscription

    for i in range(100):
        #value = subscription.read()
        # if value:
        #     print(f"Polling, timestamp: {value[0].timestamp} value: {value[0].value}")
        time.sleep(1)


    # close the subscription when done
    sub.close()


if __name__ == "__main__":
    print("cannot do anything")