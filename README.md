# Optimizing-association-rule-hiding-using-Genetic-Algorithm

an attempt to use a Genetic Algorithm in rule hiding following another MATLAB code and report as a reference

no libraries will be required outside of the main python ones

this is an attempt to recreate a MATLAB code using python . i was unable to use the code itself at this time i don't know how to use matlab nor was i able to run the code and to keep up with the deadline i used the Technical report that came with the code as my main referance.

the report suggest that if we have a database D of transactions and each transaction is a set of items Each association rule is defined as an implication of the form X=>Y where X and Y are frequent items in D.

out goal is to mask those transactions since they have some value be deleting them from the database with the minimum impact to the purity of the dataset

so this code essentially finds association rules in a dataset, identifies sensitive ones, and modifies the dataset using a genetic algorithm to hide those rules while preserving non-sensitive rules as much as possible.

note : that i used LLMs (like chatgpt) to help me in general but the most heavy help i needed was with the genetic code please keep that in mind .

the used math equations are as follows :

\*The support of an itemset is defined as fraction of transactions in the database that contain both items A and B or as in: alpha({A,B})=|A UNITED B|/m , where m is the number of transactions in database.

\*The confidence is defined as ratio of rule support over the left hand side support or as in:
beta(X=>Y)=alpha(X=>Y)/alpha(X)

the goal is to creat a new database D prime where all the all sensitive knowledge are concealed while the number of non-sensitive patterns effected by the removal is minimumized.

now we will work on one single item that we will call Victim and is defined as the Item with the lowest support from the right hand side of the sensitive association rule.

finally we have the number of transactions which need to be altered in order to hide a specific sensitive association rule as in : delta(X=>Y)=alpha(X=>Y)=floor(beta min\*aplha(X)).

remember that during the code X is refered to as lfh and Y as rhs.

now the finall part is The process of choosing the best transactions starts from choosing the transactions which the victim item belongs to. These transactions are what the genetic algorithm is going to choose from. The objective function of the genetic algorithm is minimizing the number of lost non-sensitive association rules.

i followed the pseudocode in the proposed algorithm to the best of what i could but my lack of experince before hand with any genetic algorithm made it really hard to do it by the dead line but for the most part it is the same .


=======================================================================================================================
in the new version (1.2) i made sure that the delta function is better and the results are stored in an excel file so i t can be graphed and read , i added some evaluation values too like the accurecy , it is still not perfect and there is few things i know that are wrong or requiere debugging , also the perfomance is really bad now it will not be good to run this code on any DB with more that 10k transactions and even then it's not perfect .
