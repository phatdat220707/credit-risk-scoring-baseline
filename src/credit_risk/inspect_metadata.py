from ucimlrepo import fetch_ucirepo

german_credit = fetch_ucirepo(id=144)

print(german_credit.metadata["additional_info"]["variable_info"])

print("Metadata:")
print(german_credit.metadata)

print("\nVariables:")
print(german_credit.variables)