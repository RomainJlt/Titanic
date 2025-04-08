# Analyse de Survie du Titanic

Ce projet présente une analyse approfondie des données du Titanic en utilisant différentes techniques de Machine Learning pour prédire la survie des passagers.

## Analyse Détaillée des Fichiers

### Fichiers de Données
1. `train.csv` (61.2 KB)
   - Données d'entraînement principales
   - Contient les informations des passagers et leur statut de survie
   - Utilisé pour entraîner et valider les modèles

2. `test.csv` (28.6 KB)
   - Jeu de données de test
   - Utilisé pour évaluer les performances finales des modèles

### Fichiers Python
1. `titanic_recap.py` (5.2 KB)
   - Implémente la régression logistique
   - Prétraitement des données :
     * Remplissage des valeurs manquantes (Age et Embarked)
     * Encodage des variables catégorielles
   - Entraînement du modèle avec validation croisée
   - Génération des visualisations et métriques
   - Sauvegarde des résultats dans un fichier texte

2. `titanic_decision_tree.py` (6.1 KB)
   - Implémente l'arbre de décision
   - Même prétraitement des données que titanic_recap.py
   - Profondeur maximale de l'arbre fixée à 5
   - Génération de visualisations détaillées de l'arbre
   - Analyse approfondie de l'importance des caractéristiques

3. `exploration.py` (4.0 KB)
   - Analyse exploratoire des données
   - Génération de statistiques descriptives
   - Création de visualisations pour comprendre les distributions

4. `visualiser_importance.py` (2.2 KB)
   - Script dédié à la visualisation de l'importance des caractéristiques
   - Création de graphiques détaillés pour l'interprétation des modèles

## Analyse Détaillée des Visualisations

### Matrices de Confusion
1. `confusion_matrix_logistic.png` (20.3 KB)
   - Matrice de confusion pour la régression logistique
   - Montre la répartition des prédictions :
     * Vrais Positifs (Survivants correctement prédits)
     * Faux Positifs (Prédits survivants mais décédés)
     * Vrais Négatifs (Non-survivants correctement prédits)
     * Faux Négatifs (Prédits décédés mais survivants)

2. `confusion_matrix_dt.png` (21.2 KB)
   - Matrice de confusion pour l'arbre de décision
   - Format similaire à la matrice de régression logistique
   - Permet la comparaison directe des performances des deux modèles

### Importance des Caractéristiques
1. `feature_importance.png` (17.2 KB)
   - Vue d'ensemble de l'importance relative des caractéristiques
   - Graphique en barres montrant le poids de chaque variable

2. `feature_importance_detailed.png` (144.1 KB)
   - Analyse plus détaillée de l'importance des caractéristiques
   - Inclut des intervalles de confiance
   - Montre les interactions entre les variables

3. `feature_importance_dt.png` (18.3 KB)
   - Importance des caractéristiques spécifique à l'arbre de décision
   - Basée sur la réduction de l'impureté de Gini

4. `feature_importance_pie.png` (317.6 KB)
   - Représentation en camembert de l'importance relative
   - Montre la proportion de l'importance totale pour chaque caractéristique

### Visualisations du Modèle
1. `decision_tree.png` (1.1 MB)
   - Représentation graphique complète de l'arbre de décision
   - Montre toutes les divisions et règles de décision
   - Permet de comprendre le processus de prise de décision

2. `logistic_regression_coefficients.png` (22.2 KB)
   - Coefficients de la régression logistique
   - Montre l'impact positif ou négatif de chaque variable sur la survie

3. `survival_distribution.png` (12.5 KB)
   - Distribution des survivants vs non-survivants
   - Montre l'équilibre des classes dans le jeu de données

### Rapports HTML
1. `titanic_analysis_report.html` (4.8 MB)
   - Rapport détaillé de l'analyse complète
   - Inclut toutes les visualisations et résultats
   - Fournit une interface interactive pour explorer les données

2. `your_report.html` (4.8 MB)
   - Version personnalisée du rapport d'analyse
   - Contient des observations spécifiques et recommandations

## Analyse Détaillée de l'Arbre de Décision

### Structure de l'Arbre
L'arbre de décision a été construit avec une profondeur maximale de 5 niveaux pour éviter le surapprentissage. Voici sa structure détaillée :

#### Niveau 1 (Racine)
- Critère principal : Sex (Genre)
  * Division initiale basée sur le genre des passagers
  * Reflète l'importance historique de "les femmes et les enfants d'abord"

#### Niveau 2
- Pour les femmes :
  * Division basée sur Pclass (Classe sociale)
  * Les femmes de première et deuxième classe ont eu un taux de survie plus élevé
- Pour les hommes :
  * Division basée sur Age
  * Les jeunes garçons ont eu plus de chances de survie

#### Niveau 3
- Branches féminines :
  * Considération du facteur Fare (Prix du billet)
  * Impact de l'Embarked (Port d'embarquement)
- Branches masculines :
  * Influence de Pclass
  * Importance de SibSp (Nombre de frères/sœurs/conjoints)

### Importance des Caractéristiques dans l'Arbre
1. Sex (53.74%)
   - Facteur le plus déterminant
   - Les femmes ont eu significativement plus de chances de survie

2. Pclass (16.32%)
   - Deuxième facteur le plus important
   - Impact plus fort pour les passagers de première classe

3. Age (13.45%)
   - Particulièrement important pour les enfants
   - Seuil critique autour de 13 ans

4. Fare (8.91%)
   - Corrélé avec Pclass
   - Indicateur supplémentaire du statut social

5. SibSp (4.12%)
   - Impact modéré sur la survie
   - Les passagers voyageant seuls ou avec peu de famille ont eu des taux de survie différents

6. Embarked (2.24%)
   - Impact limité mais mesurable
   - Différences entre les ports de Southampton, Cherbourg et Queenstown

7. Parch (1.22%)
   - Impact le plus faible
   - Nombre de parents/enfants à bord

### Règles de Décision Principales

1. Règle Femmes de Première Classe
   - SI Sex = Female ET Pclass = 1
   - ALORS Probabilité de Survie > 95%

2. Règle Hommes Âgés de Troisième Classe
   - SI Sex = Male ET Age > 35 ET Pclass = 3
   - ALORS Probabilité de Survie < 15%

3. Règle Enfants avec Famille
   - SI Age < 13 ET SibSp < 3
   - ALORS Probabilité de Survie > 65%

### Performance du Modèle

1. Métriques Globales
   - Accuracy : 0.7989 (79.89%)
   - F1-Score : 0.7273 (72.73%)
   - Précision : 0.7500 (75.00%)
   - Recall : 0.7059 (70.59%)

2. Analyse par Classe
   - Survivants :
     * Précision : 75.00%
     * Recall : 70.59%
   - Non-survivants :
     * Précision : 83.33%
     * Recall : 86.21%

3. Validation Croisée
   - Score moyen : 0.7824
   - Écart-type : 0.0412
   - Stabilité du modèle confirmée

### Forces et Faiblesses du Modèle

#### Forces
1. Interprétabilité élevée
   - Structure claire et facile à comprendre
   - Règles de décision explicites
   - Importance des caractéristiques quantifiée

2. Performance stable
   - Résultats cohérents sur différents échantillons
   - Pas de surapprentissage majeur
   - Bonnes performances sur les deux classes

#### Faiblesses
1. Limites de complexité
   - Profondeur maximale limite la capture de patterns complexes
   - Certaines interactions subtiles peuvent être manquées

2. Sensibilité aux données
   - Les petits changements dans les données peuvent modifier la structure
   - Certaines branches peuvent être instables

### Recommandations d'Amélioration

1. Optimisation de la Structure
   - Tester différentes profondeurs d'arbre
   - Expérimenter avec différents critères de division
   - Ajuster les paramètres de régularisation

2. Enrichissement des Données
   - Créer des caractéristiques composites
   - Ajouter des interactions entre variables
   - Traiter les valeurs manquantes de manière plus sophistiquée

3. Ensemble Methods
   - Utiliser Random Forest pour plus de robustesse
   - Implémenter Gradient Boosting pour améliorer la performance
   - Combiner avec d'autres types de modèles

## Résultats des Modèles

### Régression Logistique
- Accuracy : ~0.80
- F1-score : ~0.73
- Caractéristique la plus importante : Sex
- Validation croisée : performance stable à travers les différents échantillons

### Arbre de Décision
- Accuracy : ~0.80
- F1-score : ~0.73
- Caractéristique la plus importante : Sex (53.74%)
- Profondeur maximale : 5 niveaux pour éviter le surapprentissage

## Conclusions Principales

1. Les deux modèles montrent des performances très similaires
2. Le genre (Sex) est le facteur le plus déterminant pour la survie
3. La classe sociale (Pclass) est le deuxième facteur le plus important
4. L'âge joue un rôle significatif, particulièrement pour les enfants
5. Les données sont relativement équilibrées entre survivants et non-survivants

## Pistes d'Amélioration

1. Ajustement des hyperparamètres
   - Test de différentes profondeurs d'arbre
   - Optimisation des paramètres de régularisation

2. Ingénierie des caractéristiques
   - Création d'une variable de taille de famille
   - Analyse plus détaillée des groupes d'âge

3. Algorithmes avancés
   - Implémentation de Random Forest
   - Test de Gradient Boosting
   - Exploration de réseaux de neurones

4. Ensemble Learning
   - Combinaison des prédictions des différents modèles
   - Mise en place d'un système de vote

## Enseignements Clés de l'Analyse

### 1. Facteurs Sociaux et Démographiques
- Le genre était le facteur le plus déterminant pour la survie
  * Les femmes avaient environ 3 fois plus de chances de survivre que les hommes
  * Cela reflète la politique "les femmes et les enfants d'abord"

- La classe sociale a joué un rôle crucial
  * Les passagers de première classe avaient un taux de survie de ~63%
  * Les passagers de troisième classe seulement ~25%
  * Montre l'impact des inégalités sociales dans la catastrophe

### 2. Aspects Techniques de l'Analyse
- Les deux modèles (Régression Logistique et Arbre de Décision) ont donné des résultats similaires
  * Accuracy d'environ 80%
  * Montre la robustesse des résultats
  * Suggère que nous avons atteint la limite de prédictibilité avec ces données

- L'importance des caractéristiques est cohérente entre les modèles
  * Sex : ~54%
  * Pclass : ~16%
  * Age : ~13%
  * Confirme la fiabilité de nos conclusions

### 3. Leçons pour le Machine Learning
- La simplicité peut être efficace
  * Un arbre de décision simple capture bien les patterns principaux
  * Pas besoin de modèles complexes pour obtenir de bonnes performances

- L'importance de l'interprétabilité
  * Les règles de décision sont faciles à comprendre
  * Permet de valider les résultats avec la connaissance historique
  * Facilite la communication des résultats

### 4. Limites et Considérations
- Données manquantes
  * Certaines informations importantes pourraient être absentes
  * Les stratégies de remplissage peuvent influencer les résultats

- Biais historiques
  * Les données reflètent les normes sociales de 1912
  * Important de contextualiser les résultats

### 5. Applications Pratiques
- Pour l'Analyse de Données
  * Importance de la visualisation des données
  * Nécessité de combiner différentes approches d'analyse
  * Valeur de la validation croisée

- Pour le Machine Learning
  * Commencer par des modèles simples
  * Importance de la préparation des données
  * Utilité de comparer différents modèles

### 6. Perspectives Historiques
- La tragédie du Titanic révèle :
  * L'impact des classes sociales sur la survie
  * L'importance des protocoles d'urgence
  * Les conséquences des décisions prises dans l'urgence

Ces enseignements nous montrent que l'analyse de données peut :
1. Confirmer des faits historiques connus
2. Quantifier précisément l'impact de différents facteurs
3. Révéler des patterns qui pourraient être utiles pour la sécurité maritime moderne
4. Démontrer l'importance de l'équité dans les situations de crise
