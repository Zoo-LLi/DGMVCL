# GCL-MVC

You may reproduce the experimental results  if setting the following parameter "load_model" to True. This means that the program will load the trained model provided by the authors for clustering tasks.

parser.add_argument('--load_model', default=False, help='Testing if True or training.')

You may consult the commented code in the file “main.py” for guidance on selecting the proper parameters if the trained model is unavailable for additional datasets.

Thank you.

--Prerequisites

Linux

--Required Python Packages

python>=3.9.7

pytorch>=1.7.1

numpy>=1.21.5

scikit-learn>=1.0.1

scipy>=1.7.3

