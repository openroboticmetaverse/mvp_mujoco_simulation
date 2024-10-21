import time
import motorcortex

def message_received(parameters, simulation):
    # Update the simulation's q_list whenever a new message is received
    for cnt in range(0, len(parameters)):
        param = parameters[cnt]
        value = param.value
        q_list = list(value)
        simulation.update_q_list(q_list)  # Update q_list in the simulation

def run_motorcortex(simulation):
    """This function will handle the MotorCortex connection in a separate thread."""
    parameter_tree = motorcortex.ParameterTree()

    try:
        req, sub = motorcortex.connect("wss://192.168.56.101:5568:5567", motorcortex.MessageTypes(), parameter_tree,
                                        certificate="mcx.cert.crt", timeout_ms=1000,
                                        login="admin", password="vectioneer")
    except RuntimeError as err:
        print(err)
        return

    paths = ['root/AxesControl/axesPositionsActual']
    divider = 100
    subscription = sub.subscribe(paths, 'group1', divider)
    is_subscribed = subscription.get()
    if (is_subscribed is not None) and (is_subscribed.status == motorcortex.OK):
        print(f"Subscription successful, layout: {subscription.layout()}")
    else:
        print(f"Subscription failed, do your paths exist? \npaths: {paths}")
        sub.close()
        return

    # Set the callback to update the q_list in the MuJoCoSimulation
    subscription.notify(lambda params: message_received(params, simulation))

    # Keep the subscription running
    while True:
        time.sleep(1)

    sub.close()
