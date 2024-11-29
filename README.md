# Portfolio Selection NN Hopfield Dynamics
Welcome to the Portfolio Selection Neural Network repository! This project implements a portfolio optimization model using a Hopfield-based neural network, designed to select an optimal portfolio by balancing risk and return. The model operates under constraints such as cardinality (the number of assets to include) and bounding constraints (upper and lower bounds for capital allocation in each asset). The portfolio is optimized through an energy minimization process guided by the Hopfield network dynamics. The goal of this project is to provide a practical, high-performance implementation of portfolio selection, bridging the gap between theoretical concepts and real-world applications.

## Table of Contents
- [Requirements](#requirements)
- [Installation and Usage](#installation-and-usage)
- [Contributors](#contributors)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Requirements
- Python 3.X.X

## Installation and Usage
Clone the repository and navigate into its directory:
   
 ```sh
 git clone https://github.com/rorro6787/portfolio-selection-nn-hopfield-dynamics.git
 cd portfolio-selection-nn-hopfield-dynamics
 ```
Install dependencies and run the training/testing script:
 ```sh
 chmod +x setup.sh
 ./setup.sh
 ```

## Contributors
- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/rorro6787) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/emilio-rodrigo-carreira-villalta-2a62aa250/) **Emilio Rodrigo Carreira Villalta**

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## Acknowledgements
Inspired by various tutorials and resources on neural networks and the following article: "Portfolio selection using neural networks:" [Read the Article](https://github.com/rorro6787/portfolio-selection-nn-hopfield-dynamics/blob/main/PortfolioHopfield.pdf).