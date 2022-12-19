# Final Year Project
This is a repository for all the files relating to my FYP project

HOW TO SETUP PROJECT ENVIRONMENT
-------------------------------
To setup this project, you must have Anaconda, Google Chrome, Git and Heroku CLI installed on your machine

Open Anaconda and import the enviroment via the import button within the 'Environments':
Import from Local drive, select fyp conda environment.yaml which can be found within this folder

If environment failed to load from then please follow the steps below
Open Anaconda and install the enviroment with the command below:
$ conda create --name <env> --file requirments.txt
The required 3rd party packages required for Anaconda can be found in the requirments.txt file within this folder

HOW TO RUN MODEL CREATION/TESTING FOR USAGE IN HEROKU
-----------------------------------------------------
Launch main.py from Anaconda in the 'Model Creation' folder to perform the preprocessing of the datatset and creation of the model


HOW TO UPLOAD HEROKU APP TO HEROKU
----------------------------------
Requirments for the Heroku App can be found in requirments.txt from the 'Herkou Deployment' folder
Run the following commands from the Heroku CLI inside of the 'Heroku Deployment' folder to create/deploy the app to Heroku

$ heroku login
login with a Heorku account
$ git init
$ git add .
$ git commit -m "1"

Useful page if you are struggling to deploy Heroku app https://devcenter.heroku.com/articles/git
If you cannot upload your own version of the app then please find the working version at https://heroku-deployment-fyp.herokuapp.com/


HOW TO INSTALL/USE THE EXTENSION
--------------------------------
Install the extension by going to manage extension in Google Chrome, then click on 'Load Unpacked' then select the 'Gmail Detect Phishing Extension' as the target and click open
The extension is now installed on Google Chrome
Pin the extension on Google Chrome so that it can be accessed more easily
To use the extension, open Gmail.com and login with an account, select an email to scan, then highlight the body of the email, right click on it and select 'Copy'
Now that the body of the email has been copied, open the extension by clicking on the extension icon, paste the copied text into the form and then click the submit button
The extension will now determine if it is a phishing by finding certain key attributes from the selected text
You should now see the result page which tells you if the selected text was a phishing attempt or not

Please test the extension by using one of the Phishing values found within the phishing_data_by_type.csv file in the Dataset folder
