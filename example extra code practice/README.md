## 20201008
这是一节practice课的练习，具体的要求如下：
### Task 1: Duplicate Removal

Write a function that takes as its input a list, and creates a new list with all of the duplicate entries removed. Then print out the new list. You should use a for loop in your solution. 

Try your function on the following lists:

L = [3,7,11,-2,7,-7,1,13,7,2,3,1]

L = ["hi","Rob","Steven","hello","hello world","hi world","world","Rob","hello","hello Steven"]

L = [2,"hi",-2,"2",0,"=","Rob","0",2.0,0]

非常简单，python自带的set（集合）是天然去重的，转换成set再转出来list就可以了

另一种思路则是，构建一个空列表，然后用生成式把空列表里没有的元素给它塞进去

### Task 2: The Guessing Game

Generate a random number from 1 to 99, and ask the user to input a number. Your program should tell the player if their guess is "too high", "too low" or "correct". If the player guesses too high or too low, the program should ask them to input another number, and again tell the player if they are too high, too low or correct. Keep doing this until the player inputs the correct number. When the player inputs the correct number, you should print out a message saying how many guesses the player had.

You will need to remember or look up how to generate a random number in the range 1 to 99, and you will also need some code to ask the user to input a value. You can do this with the code:

x = input("Enter a number:")

A box will appear under your codebox, with the message "Enter a number:". When you enter a number and press enter, that value will be passed to the variable x. 

非常简单的循环，一个while，然后几个if判断条件就结束了

### Task 3: Prime Factorisation

Ask the user for an input number, and output the prime factorisation of that number. E.g. If the user inputs the number 12, the output should be 12 = 2 * 2 * 3. If the user inputs the number 32, the output should be 32 = 2 * 2 * 2 * 2 * 2. If the user enters the number 30, the output should be 30 = 2 * 3 * 5. The user should be able to enter any number up to 100,000,000. Use a line of code similar to that above to ask the user for an input number.

稍微有些麻烦的问题，关键问题在于拆分步骤。首先，要回忆起因式分解我们是怎么做的——x，找到它的一个除数，2、3、5……——除过去，再拿就到的数迭代进去继续找下一个除数。

所以这个问题的解法就是，先构造一个列表用于储存因子，然后开始作法——先试探2是不是，不行就3，不行继续……找到这个因子之后，除掉它，然后继续……直到你最后搞到了1（也就是最后一个因子找到了）