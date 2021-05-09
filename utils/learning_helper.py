def epoch_time(start_time, end_time):
    """
    Compute the time spent for each epoch
    :param start_time: the time before each epoch
    :param end_time: the time after each epoch
    :return: the time spent for each epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
