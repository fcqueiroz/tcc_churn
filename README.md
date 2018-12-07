# Bachelor Final Project
### Applying Decision Trees for predicting Client Churn in Non-Contractual Settings

This work is being developed as a bachelor final project and intends to address 
the churn problem in non-contractual settings using machine learning techniques. 
The main goals are:
* Find out if we can generate better features exploring the network effects of 
[two-sided markets](https://en.wikipedia.org/wiki/Two-sided_market)
* Find a churn indicator more relevant for the business than the current heuristic 
rule 
* Find relevant criteria for predicting when a valuable client will churn  
  
## Getting Started

### Prerequisites
* Docker

#### Preparing the environment

Clone this repository in the current folder  
`git clone https://github.com/fcqueiroz/tcc_churn.git`  

Get inside the project root folder  
`cd tcc_churn`  

Build the docker image  
`docker build --no-cache -t tcc_churn .`

Please check the instructions provided in [make_dataset.py](src/data/make_dataset.py)
to generate your own dataset before running the program.

#### Using Jupyter Notebook
Substitute HOSTPORT by the desired host port that will bind to the container 
(eg 8888)  

`docker run -d -v /PATH/TO/tcc_churn:/tcc_churn --name notebook -p HOSTPORT:8888 tcc_churn jupyter notebook`

#### Training, Testing and Saving the model

In all the following examples, substitute `/PATH/TO/tcc_churn` by the complete path 
to reach the project root folder eg `/home/fernanda/tcc_churn` or 
`C:\User\fernanda\tcc_churn` 

__Interactive mode__  
The program can be run in interactive mode with the following command:  

`docker run -i -v /PATH/TO/tcc_churn:/tcc_churn tcc_churn python -m src.base`

__Batch / Non interactive mode__  
Windows users running docker in cmd.exe might face problems when passing commands 
in the interactive mode for them not being recognized. An easy way to deal with 
that is using the batch mode and passing all the commands at once when running 
the script.  

The available commands (case insensitive) are:
* train: Train model (and transform raw data in processed datasets if needed)
* test: Test model performance
* load: Load trained model and processed datasets
* save: Save trained model and processed datasets
* exit: Exit program

For example, the following command will read the raw data, transform in a suitable
shape for the machine learning algorithm, test its performance in unseen data,
save the model and the processed datasets for later use and gracefully exit the program.  

`docker run -i -v /PATH/TO/tcc_churn:/tcc_churn tcc_churn python -m src.base train test save exit`

The testing performance will be saved in [reports/performance.csv](reports/performance.csv)

## Known Issues
* There is no available dataset for running the project. 
* Base script is silly. 

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Special thanks to [Fernando C Gomide](http://www.dca.fee.unicamp.br/~gomide/) for providing directions for how to start this project and for accepting being my guiding teacher.
* [Thiago Oliveira](https://www.linkedin.com/in/thiagosoaresdeoliveira/) for the inspiration in pursuing this business problem and for many useful questions.
* [Cauê Polimanti](https://github.com/CaueP) for helping me fill some knowledge holes in software engineering and OO programming.