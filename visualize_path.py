import sys
import matplotlib.pyplot as plt

# Takes in a single command line argument, which is file name with path data
def main():
    filename = sys.argv[1]

    with open(filename) as f:
        data = f.read()

    data = data.split('\n')

    x = [row.split(' ')[0] for row in data]
    y = [row.split(' ')[1] for row in data]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.set_title("Constrained Traveling Salesman Path through California")    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    plt.plot(x, y, color="blue", linestyle="-")

    plt.show()


if __name__ == '__main__':
    main()