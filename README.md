# DGMVCL

This project is about Grassmannian Contrastive Learning-based Multi-view Subspace Clustering and aims to address the issues related to it. It has been enhanced and improved upon the foundation of CVCL to enhance its performance and functionality.

You may reproduce the experimental results  if setting the following parameter "load_model" to True. This means that the program will load the trained model provided by the authors for clustering tasks.

parser.add_argument('--load_model', default=False, help='Testing if True or training.')

You may consult the commented code in the file “main.py” for guidance on selecting the proper parameters if the trained model is unavailable for additional datasets.

Thank you.


--Required Python Packages

python>=3.9.7

pytorch>=1.7.1

numpy>=1.21.5

scikit-learn>=1.0.1

scipy>=1.7.3

If you have any questions or suggestions, please feel free to reach out to us at any time!
