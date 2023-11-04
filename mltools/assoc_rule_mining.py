from itertools import combinations
from collections import defaultdict
import argparse

class AssociationRuleMining:
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = None
        self.supports = None
        self.rules = None

    def generate_itemsets(self, transactions):
        itemsets = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                itemsets[frozenset([item])] += 1
        return itemsets

    def filter_itemsets(self, itemsets, num_transactions):
        filtered_itemsets = {}
        for itemset, support in itemsets.items():
            if support / num_transactions >= self.min_support:
                filtered_itemsets[itemset] = support
        return filtered_itemsets

    def generate_candidate_itemsets(self, itemsets, k):
        candidate_itemsets = set()
        for itemset1 in itemsets:
            for itemset2 in itemsets:
                if len(itemset1.union(itemset2)) == k:
                    candidate_itemsets.add(itemset1.union(itemset2))
        return candidate_itemsets

    def prune_itemsets(self, candidate_itemsets, prev_itemsets):
        pruned_itemsets = set()
        for itemset in candidate_itemsets:
            subsets = list(combinations(itemset, len(itemset) - 1))
            is_valid = True
            for subset in subsets:
                if frozenset(subset) not in prev_itemsets:
                    is_valid = False
                    break
            if is_valid:
                pruned_itemsets.add(itemset)
        return pruned_itemsets

    def generate_association_rules(self, itemsets):
        association_rules = []
        for itemset in itemsets:
            if len(itemset) >= 2:
                for i in range(1, len(itemset)):
                    antecedent = frozenset(itemset[:i])
                    consequent = frozenset(itemset[i:])
                    confidence = itemsets[itemset] / itemsets[antecedent]
                    if confidence >= self.min_confidence:
                        association_rules.append((antecedent, consequent, confidence))
        return association_rules

    def fit(self, transactions):
        num_transactions = len(transactions)
        self.itemsets = self.generate_itemsets(transactions)
        k = 2
        self.supports = {}
        self.supports[1] = self.filter_itemsets(self.itemsets, num_transactions)
        while len(self.supports[k-1]) > 0:
            candidate_itemsets = self.generate_candidate_itemsets(self.supports[k-1], k)
            candidate_itemsets = self.prune_itemsets(candidate_itemsets, self.supports[k-1])
            itemsets_k = defaultdict(int)
            for transaction in transactions:
                for candidate_itemset in candidate_itemsets:
                    if candidate_itemset.issubset(transaction):
                        itemsets_k[candidate_itemset] += 1
            self.supports[k] = self.filter_itemsets(itemsets_k, num_transactions)
            k += 1
        self.rules = self.generate_association_rules(self.itemsets)

    def display_rules(self):
        for antecedent, consequent, confidence in self.rules:
            print(f"Rule: {antecedent} => {consequent}, Confidence: {confidence:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Association Rule Mining')
    parser.add_argument('--min_support', type=float, default=0.1, help='Minimum support threshold (default is 0.1)')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Minimum confidence threshold (default is 0.5)')
    args = parser.parse_args()

    # Example usage of the AssociationRuleMining class:
    transactions = [
        {'apple', 'banana', 'cherry'},
        {'banana', 'cherry'},
        {'apple', 'banana'},
        {'apple', 'cherry'},
        {'apple', 'banana', 'cherry'},
        {'banana'},
        {'cherry'}
    ]
    association_miner = AssociationRuleMining(min_support=args.min_support, min_confidence=args.min_confidence)
    association_miner.fit(transactions)
    association_miner.display_rules()

if __name__ == "__main__":
    main()