def create_output_string(joint_names, name_prefix, joint_values):
    """
    Taking two arrays and generate the required output format.
    name_prefix input is put before the joint_name.
    """

    if len(joint_names) != len(joint_values):
        raise ValueError("Input lists must have the same length")

    joint_positions = {
        joint_name: joint_value
        for joint_name, joint_value in zip(joint_names, joint_values)
    }

    output_string = '{"jointPositions":{'
    output_string += ",".join(
        f'"{name_prefix}{joint_name}":{joint_value}' for joint_name, joint_value in joint_positions.items()
    )
    output_string += "}}"

    return output_string






def test_function():
    joint_names_test = ["panda_joint1", "panda_joint2", "panda_joint4", "panda_joint6", "panda_joint7"]
    joint_values_test = [0.8326047832305474, 1.4897112838523152, -1.3823714659725588, -0.9357293466958847, 0.7023423271030382]
    
    output = create_output_string(joint_names_test, joint_values_test)
    print(output)


if __name__ == "__main__":
    test_function()