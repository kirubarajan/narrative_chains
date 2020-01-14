#$ -N narrative-chains
#$ -o narrative.stdout
#$ -e narrative.stderr
#$ -cwd
#$ -V
source venv/bin/activate
python src/index.py
