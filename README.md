# ADM-HW#3
*Madina Rakhimbayeva 2189800

![iStock-654454404-777x518](https://a.storyblok.com/f/125576/2448x1220/327bb24d32/hero_update_michelin.jpg/m/1224x0/filters:format(webp))


## Project Overview
This repository contains the submission for the [third homework](https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_3) of the ADM course. 

### Search Engine
The main project implements a search engine for [Michelin-starred restaurants in Italy](https://guide.michelin.com/en/it/restaurants). The objective is to design and develop both conjunctive and ranked search engines, enabling users to search for specific restaurants based on their preferences. This search engine leverages TF-IDF and Cosine Similarity to process and rank search results, allowing efficient data retrieval and accurate ranking.

### Algorithm Question
The task is to determine if a robot can collect all packages on a grid using the shortest path while moving only up and right, prioritizing lexicographical order in case of ties, and to analyze the algorithm's correctness and complexity. Additionally, we evaluate the greedy approach when the robot can move in all directions, assessing its optimality or providing counterexamples.

## Structure
This repository contains the following files:

- **`main.ipynb`**: The Jupyter Notebook containing the complete resolution of Homework 3.  
- **`data_collection.py`**: Includes all the functions used for solving the **Data Collection** section (Point 1).  
- **`search_engine.py`**: Contains all the functions used for solving the **Search Engine** section (Point 2).  
- **`new_score.py`**: Includes all the functions used for solving the **New Score** section (Point 3).  
- **`restaurants_by_region_map.html`**, **`restaurants_map.html`**, **`top_10_restaurants_map.html`**: HTML files used for **Visualizing the Most Relevant Restaurants** in Section 4.  
- **`city_region_coordinates.csv`**: Contains uncleaned data of the coordinates used in Section 4.  
- **`city_region_coordinates_cleaned.csv`**: Contains cleaned data of the coordinates used in Section 4.  
- **`tfIdf_inverted_index.json`**, **`inverted_index.json`**: JSON files containing key-value pairs for the dictionaries used in Section 2.  
- **`vocabulary.csv`**: A file mapping all words in the descriptions after preprocessing.  
- **`algorithmic_question.py`**: Includes the code for the algorithm required in the **Algorithmic Question** problem.  
- **`input_algorithmic_question.txt`**: Contains the input data for the algorithm in the **Algorithmic Question** section.


## How to Use

You can view the notebook by opening `main.ipynb` in any Jupyter notebook environment or through GitHub's integrated [notebook viewer].   
To run the code, ensure to copy all the content of the repository. 
