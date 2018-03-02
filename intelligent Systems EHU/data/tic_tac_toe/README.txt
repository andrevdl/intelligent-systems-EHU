1. Title: Tic-Tac-Toe Endgame database

2. Source Information
   -- Creator: David W. Aha (aha@cs.jhu.edu)
   -- Donor: David W. Aha (aha@cs.jhu.edu)
   -- Date: 19 August 1991

3. Relevant Information:

   This database encodes possible board configurations
   at the end of tic-tac-toe games, where "x" is assumed to have played
   first.  The target concept is "win for x" (i.e., true when "x" has one
   of 8 possible ways to create a "three-in-a-row").  

4. Number of Instances: 574 in training, 384 in test 

5. Number of Attributes: 9, each corresponding to one tic-tac-toe square

6. Attribute Information: (x=player x has taken, o=player o has taken, b=blank)

    1. top-left-square: {x,o,b} {2,1,0}
    2. top-middle-square: {x,o,b} {2,1,0}
    3. top-right-square: {x,o,b} {2,1,0}
    4. middle-left-square: {x,o,b} {2,1,0}
    5. middle-middle-square: {x,o,b} {2,1,0}
    6. middle-right-square: {x,o,b} {2,1,0}
    7. bottom-left-square: {x,o,b} {2,1,0}
    8. bottom-middle-square: {x,o,b} {2,1,0}
    9. bottom-right-square: {x,o,b} {2,1,0}
   10. Class: Class: {positive=1 (i.e., win_for_x), negative=0 (i.e., draw or loss)}
 
