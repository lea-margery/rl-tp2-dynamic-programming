# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    # a = 1
    # BEGIN SOLUTION
    memo = {}
    def aux(n: int, memo: list[int]) -> int:
        if n <= 0:
            memo[n] = 1
        if n % 2 == 1:
            memo[n] = 0
        if n not in memo: 
            memo[n] = 4 * domino_paving(n-2) - domino_paving(n-4)
        return memo[n]

    return aux(n, memo)
    # END SOLUTION


