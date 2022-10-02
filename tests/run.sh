EXIT_STATUS=0

echo "Running flake8..."
flake8 ../tests|| EXIT_STATUS=$?
flake8 ../sklearn_prod/python|| EXIT_STATUS=$?
echo "Finished!\n"

echo "Running mypy..."
mypy ../tests|| EXIT_STATUS=$?
mypy ../sklearn_prod/python|| EXIT_STATUS=$?
echo "Finished!\n"

echo "Exit status code $EXIT_STATUS!" 
exit $EXIT_STATUS
