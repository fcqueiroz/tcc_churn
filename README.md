# Bachelor Final Project
### Applying Decision Trees for predicting Client Churn in Non-Contractual Settings

This work is being developed as a bachelor final project and intends to address the churn problem in non-contractual settings using machine learning techniques. The main goals are:
* Find out if we can generate better features exploring the network effects of [two-sided markets](https://en.wikipedia.org/wiki/Two-sided_market)
* Find a churn indicator more relevant for the business than the heuristic rule 
* Find relevant criteria for predicting when a valuable client will churn  
  
## Getting Started

#### Using Docker

Clone this repository in the current folder  
`git clone https://github.com/fcqueiroz/tcc_churn.git`  

Get inside the project root folder  
`cd tcc_churn`  

Build the docker image  
`docker build -t tcc_churn:0.1 .`

Substitute `/PATH/TO` by the complete path to reach the project root folder eg `/home/fernanda/tcc_churn` or `C:\\User\fernanda\tcc_churn` and start the container  
`docker run -v /PATH/TO/data:/tcc_churn/data -v /PATH/TO/models:/tcc_churn/models -v /PATH/TO/reports:/tcc_churn/reports tcc_churn:0.1`


These instructions will execute the whole data pipeline and export the performance results to [reports/performance.csv](reports/performance.csv)

### Prerequisites
* Docker

## Known Issues
* The current release doesn't provide much flexibility
* The source code is copied inside the docker image, which might prevent docker from seeing changes in the source code besides making the image unnecessarily big. Delete the cached layers and rebuild the image after editing anything in the project
* There isn't an available dataset for running the project. A toy dataset will be provided in later releases. 
## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Special thanks to [Fernando C Gomide](http://www.dca.fee.unicamp.br/~gomide/) for providing directions for how to start this project and for accepting being my guiding teacher.
* [Thiago Oliveira](https://www.linkedin.com/in/thiagosoaresdeoliveira/) for the inspiration in pursuing this business problem and for many useful questions.
* [Cauê Polimanti](https://github.com/CaueP) for helping me fill some knowledge holes in software engineering and OO programming.

