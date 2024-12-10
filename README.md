[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/VuODydzp)
# Group members
### *Yu-hsiang Tang (jtang3) | *Chi-Yeh Chen (chiyehc)

## Dataset Features Description

This dataset contains detailed information about soccer players, including their personal details, attributes, club associations, and skills. Below is an overview of the key features in the dataset:

## Player Information
- **sofifa_id**: Unique identifier for each player.
- **player_url**: URL to the player's profile page.
- **short_name**: Player's abbreviated name.
- **long_name**: Player's full name.
- **dob**: Date of birth of the player.
- **age**: Age of the player.
- **height_cm**: Height of the player in centimeters.
- **weight_kg**: Weight of the player in kilograms.
- **nationality_id**: Unique identifier for the player's nationality.
- **nationality_name**: Name of the player's nationality.

## Club Information
- **club_team_id**: Unique identifier for the player's club.
- **club_name**: Name of the club the player is associated with. 
- **league_name**: Name of the league the club is part of.
- **league_level**: Level of the league in which the club competes.
- **club_position**: Player's position in the club (e.g., Goalkeeper, Midfielder).
- **club_jersey_number**: Jersey number assigned to the player at the club.
- **club_loaned_from**: Indicates if the player is loaned from another club.
- **club_joined**: Date when the player joined the club.
- **club_contract_valid_until**: Year until which the player's club contract is valid.

## National Team Information
- **nation_team_id**: Unique identifier for the player's national team.
- **nation_position**: Player's position in the national team.
- **nation_jersey_number**: Jersey number assigned to the player in the national team.

## Player Attributes and Skills
- **overall**: Player's overall rating.
- **potential**: Player's potential rating.
- **value_eur**: Player's market value in Euros.
- **wage_eur**: Player's weekly wage in Euros.
- **preferred_foot**: Indicates the player's dominant foot (left or right).
- **weak_foot**: Skill rating for the player's weaker foot (1 to 5).
- **skill_moves**: Skill rating indicating the player's ability to perform complex moves (1 to 5).
- **international_reputation**: Rating representing the player's global reputation (1 to 5).
- **work_rate**: Description of the player's work rate in attack and defense.
- **body_type**: Description of the player's body type (e.g., Lean, Average).
- **real_face**: Indicates if the player has a real-life face scan in the game.

## Player Performance Attributes
- **pace**: Speed rating of the player.
- **shooting**: Shooting skill rating.
- **passing**: Passing skill rating.
- **dribbling**: Dribbling skill rating.
- **defending**: Defending skill rating.
- **physic**: Physicality rating.

## Detailed Performance Metrics
- **attacking_crossing**: Player's crossing ability.
- **attacking_finishing**: Player's finishing ability.
- **attacking_heading_accuracy**: Player's heading accuracy.
- **attacking_short_passing**: Player's short passing ability.
- **attacking_volleys**: Player's volleying skill.

- *(Other attributes follow a similar pattern, including skills like dribbling, shooting power, movement speed, mentality, defending, and goalkeeping.)*

## Additional Features
- **player_positions**: Positions the player can play.
- **player_tags**: Tags associated with the player (e.g., "Leader").
- **player_traits**: Specific traits of the player (e.g., "Speed Dribbler").
- **release_clause_eur**: Release clause value in Euros.
- **player_face_url**: URL to the player's face image.
- **club_logo_url**: URL to the club's logo image.
- **club_flag_url**: URL to the club's flag image.
- **nation_logo_url**: URL to the national team's logo.
- **nation_flag_url**: URL to the national team's flag.

This list covers the main columns in the dataset. Each column provides specific information that can be used for various analyses, such as player performance evaluation, market value estimation, and team compositions.
## File Description
1. FIFA_dataset: the FIFA dataset from kaggle. It contains several csv file of female players information from 2016-2022 and male palyers information from 2015-2022.
2. chiyehc_group_project.ipynb: The main script of this project.
3. create_players_table.sql: A PostgreSQL file which is used to generate postgres schema and table.

## Discussion: Benefit of using PostgreSQL DB vs NoSQL database
This dataset has a well-defined structure with fixed columns (e.g., sofifa_id, short_name, overall, potential, etc.), each with specified data types. This is characteristic of structured data, making it more suitable for storage in a relational (SQL) database.
NoSQL databases are more suitable for handling unstructured or semi-structured data, such as documents, JSON, BSON, graphs, or when the data structure is frequently changing. If your dataset were more flexible, dynamic, or contained large amounts of unstructured information (e.g., social media data, sensor data), then a NoSQL database might be a better choice.

# Instruction of Running
## *Task1 & Task2:  Running the group_project_checkpoint1.ipynb*

### Task 1: Build and populate necessary tables 
1. Change the jar file path to make sure you can run with jdbc in cell 2.
2. Change the data path in cell 4.
3. Modify the jdbc and postgres configuration in cell 7.
4. Run the "create_players_table.sql" file in your PgAdmin query tool to create schema and table.
5. Then, run all the cell above "Task 2" markdown cell.
### Task 2: Conduct analytics on your dataset
1. Make sure you successfully run all the task 1 cells.
2. Use the "highest_contract_players" function for the 1 sub-task. (sub-task 1: In Year X, what were the Y clubs that had the highest number of players with contracts ending in year Z (or after))
3. Use the "average_age_clubs" function for the 2 sub-task. (sub-task 2: List the X clubs with the highest (or lowest) average player age for a given year Y.)
4. Run the last cell in task 2 to get the most popular nationality of each year.
   
## *Task3: Running the group_project_checkpoint2.ipynb*

### Task 3: ML Modeling (Predicting the overall ability of players)
1. Change the jar file path to make sure you can run with jdbc
2. Modify the jdbc and postgres configuration
3. Two methods to read the data:
    - Postgres
    - Convert into pandas, saving as a csv file, and then read it back.
    - Remember to modify the data path
4. Then, run all the cell in the file
### Task3 Pipeline and Results
1. **Data Cleaning (Dealing with missing value):**
    - Thresholding: drop the columns that contains over 50% missing value
    - Interpolate the missing values with average values
2. **Data Engineering:**
    - Feature selection: Select the meaningful columns as features
    - Dealing with some string data types
    - Outliers removal
    - Features correlation plot
    - Train_test split (0.8,0.2)
    - Using pipeline to preprocess the data and convert features into vectors
3. **Training:**
    - PySpark:
        - Linear regression: maxIter=100.   Train MSE: 3.42 | Test MSE: 3.43
        - Decision trees: maxDepth=12.      Train MSE: 0.90 | Test MSE: 1.19
    - Pytorch:
        - MLP: 
            - Settings: Hidden layers=3 | Neuron numbers=128 | Learning rate=1e-3 | batch size=256 | epoch=100 | optimizer=Adam
            - Train MSE: 49.9 | Test MSE: 52.0
        - MLP
            - Settings: Hidden layers=8 | Neuron numbers=512 | Learning rate=1e-3 | batch size=256 | epoch=100 | optimizer=Adam
            - Train MSE: 49.9 | Test MSE: 52.0

# Deploy to the Cloud
The group_project_checkpoint2-preprocessing.py, group_project_checkpoint2-sparkml.py, and group_project_checkpoint2-pytorch.py is the revised python script of our group_porject_checkpoint2.ipynb. These cloud version files can only be run on cloud environment.
You should run the preprocessing script to get the processed dataframe--final_df.csv. Then you can run sparkml or pytorch script in arbitary order. Those two training scripts will read final_df.csv. You can refer the screenshots in the cloud_deployment_screenshot folder to check the output which was produced when we ran on the GCP.

# Video link
[Demo1](https://youtu.be/qJo4vDwdpUs)
[Demo2](https://youtu.be/PnbWNmHdxeo)
[Demo3](https://youtu.be/gxE_Sm0zRBM)
