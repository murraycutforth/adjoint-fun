docker run -it --rm -v /home/murray/Projects/The-Pete-Equation:/home/murray/Projects/The-Pete-Equation -w /home/murray/Projects/The-Pete-Equation -e PYTHONPATH="/home/murray/Projects/The-Pete-Equation" --entrypoint "/bin/bash" quay.io/dolfinadjoint/pyadjoint:2019.1.0 -c "python3 -m unittest discover -v -s test"
