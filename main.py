# main.py
import sys
from analyzer import analyze_stock, find_buys

if __name__ == "__main__":

    arg = sys.argv[1].upper()

    if arg == "SP500":
        find_buys()
    else:
        analyze_stock(arg)
