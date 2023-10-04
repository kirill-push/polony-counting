import subprocess


def main():
    # Executing the first command: poetry run black .
    subprocess.run(["poetry", "run", "black", "."], check=True)

    # Executing the second command: poetry run isort .
    subprocess.run(["poetry", "run", "isort", "."], check=True)


if __name__ == "__main__":
    main()
