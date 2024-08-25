import numpy as np

def sudoku_solver(sudoku):
# The valid_move function determines if a value is a valid to enter into the sudoku by returning either true or false.
    def valid_move(y,x,value) :
# The function calls three different functions row, column and threebythree.
        if row(y,x,value):
# Column function checks that the number you are trying to insert is not already in that column.
            if column(y,x,value):
                if not threebythree(y,x,value):
                    return True


    def row(y,x,value):
# For loop goes through row by going through the same index of each column.
        for i in range (9) :
# If theres already the number you are trying to insert in the row then return false as the number isnt valid.
            if sudoku[y,i] == value:
                return False
        return True
        
    def column(y,x,value):
# For loops goes through columns by going through each row of the same index.
        for i in range (9) :
# If theres already the number you are trying to insert in the column then return false as the number isnt valid.
            if sudoku[i,x] == value:
                return False
        return True
       
        
    def threebythree(y,x,value):
# If the calc is equal to 0 you can start at the first index.
        if (x//3) == 0:
            for i in range (3):
                for j in range (3):
# When iterating through sub grid if the number you are trying to insert is found return false.
                    if sudoku[i,j] == value:
                        return False  
# If the calc is equal to 1 you can start at the second sub box at index 3 and increment from there.
        elif (x//3) == 1:
            for i in range (3):
                for j in range (3):
# Iterate through the whole of the second grid.
                    if sudoku[i+3,j+3] == value:
                        return False  
    
# If the calc is equal to 2 you can start at the third sub grid at index 6 and increment from there.
        elif (x//3) == 2:
            for i in range (3):
                for j in range (3):
# If number about to be inserted is found then return false.
                    if sudoku[i+6,j+6] == value:
                        return False  
  
 
    def unsolveable_sudoku():
# If the sudoku is unsolveable then a 9x9 grid full of '-1's is returned instead.
        for i in range(9):
            for j in range(9):
                sudoku[i,j] = -1
# Once this is done sudoku is returned.
        return True

# This function is to check that a sudoku isnt already correct. 
    def already_correct():
        for i in range(9):
            for j in range(9):
#If there are no 0s in the grid then its already solved so just return the sudoku.
                if sudoku[i,j] == 0:
                    return False
        return sudoku
   

    def solution() :
# The two for loops for x and y allow each position in the grid to be accessed.
        for y in range (9):
            for x in range (9):
# If a position currently has a 0 in it then it needs to be replaced hence the following lines of code.
                if sudoku[y,x] == 0 :
                    for insert in range (1,10):
# The function valid_move is called for each potential value and if the function returns true then the value is inserted.
                        if valid_move(y,x,insert) == True:
                            sudoku[y,x] = insert
# Then the function calls itself and if it always returns true it means the sudoku has been solved.
                            if solution() == True:
                                return True
    
# Otherwise the function back tracks by setting the value back to zero and trying another value to see if that works.
                            sudoku[y,x] = 0      
                    return False
        return True

    
#If sudoku solveable then return soduku
    if solution():
        return sudoku
    else:
# Otherwise call unsolveable sudoku which will return a grid of all -1s.
        unsolveable_sudoku()
        return sudoku
    

if __name__ == "__main__":
    # Example Sudoku puzzle (0s represent empty cells)
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    solution = sudoku_solver(puzzle)
    print("Sudoku solution:")
    print(solution)