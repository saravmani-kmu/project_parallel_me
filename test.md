Creaet vscode extension for sqlserver which will do perfromance tuning on sql server and provide suggestions also. 

But first step we focus on Stored Procedure performance tuning and optimization. 

This extension will identify the list of procedure names and parameters from user input and connect to sql server and fetch find the procedures and execute it 2 times (this numbers can be configurable). and take the average time and note down

Then will read the procedure code from sql server and analyze the procedure and fine tune it and execute it 2 times (this numbers can be configurable). and take the average time and note down

Then will compare the average time of original procedure and fine tuned procedure and provide the suggestions for optimization. 

Modify the proc name *_tuned and deploy in sql server
---
LLM Tasks
Identify user input and if it not enough to to find the procedure details then it should loop and ask and get proc and parameters from user

Assemtion : Already procedure is avilable in db
for now hard code the db connection string
Server=localhost\MSSQLSERVER01;Database=master;Trusted_Connection=True;

----

For testing purpose you can create test project with mock or inmemory database and create sample procedures and test it.


----------

do  u have any queries or clarification ask me before proceeding
