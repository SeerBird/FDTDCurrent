from sys import stdout


def printProgressBar (iteration, total,prefix = 'Stepping progress:', suffix = '', decimals = 1, length = 100, fill = '█'):
    total-=1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = "\n" if iteration==total else "")