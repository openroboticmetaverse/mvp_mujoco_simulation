def get_coordinates(n):
    x, y = 0, 0  # Start at the origin
    dx, dy = 1, 0  # Initial direction (right)
    segment_length = 1  # Initial segment length
    steps = 0  # Number of steps taken in the current segment
    segments_passed = 0  # Number of segments passed
    coordinates = []
    for i in range(n):
        coordinates.append([x, y, 0])

        x, y = x + dx, y + dy  # Move to the next point
        steps += 1

        if steps == segment_length:
            steps = 0
            dx, dy = -dy, dx  # Change direction anti-clockwise
            segments_passed += 1

            if segments_passed % 2 == 0:
                segment_length += 1  # Increase segment length every two segments
    return coordinates
