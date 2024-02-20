import pandas as pd
import numpy as np

class Data:
    def __init__(self,movies_file,ratings_file,users_file):
        self.movies_file = movies_file
        self.ratings_file = ratings_file
        self.users_file = users_file
        self.movies()
        self.users()
        self.ratings()
        self.generate_data_matrix()
        self.calculate_user_bias()
        self.calculate_movie_weights()
        self.calculate_unbiased_data_matrix()

    def calculate_unbiased_data_matrix(self):
        self.unbiased_data_matrix = self.data_matrix
        for i in self.data_matrix.columns:
            self.unbiased_data_matrix[i] = self.data_matrix[i] - self.bias['Bias'][i]
    
    def calculate_movie_weights(self):
        pass

    def calculate_user_bias(self):
        bias_dict = {}
        for column in self.data_matrix.columns:
            non_nan_values = self.data_matrix[column].dropna()
            num = len(non_nan_values)
            rating_sum = non_nan_values.sum()
            bias_dict[column] = rating_sum/num
        
        self.bias = pd.DataFrame.from_dict(bias_dict, orient='index', columns=['Bias'])

    def generate_data_matrix(self):
        # Generate data matrix
        self.data_matrix = self.ratings.pivot(index='MovieID',columns='UserID',values='Rating')
        self.data_matrix = self.data_matrix.fillna(np.nan)

    def movies(self):
        # Extract movies data
        with open(self.movies_file,'rb') as f:
            data=[]
            for line in f.readlines():
                line = line.decode('unicode_escape')
                line = line.strip()
                line = line.split('::')
                data.append([line[0],line[1],line[2].split('|')])
            self.movies = pd.DataFrame(data,columns=['MovieID','Title','Genres'])
            self.movies['MovieID'] = self.movies['MovieID'].astype(int)

    def ratings(self):
        # Extract ratings data
        with open(self.ratings_file,'rb') as f:
            data=[]
            for line in f.readlines():
                line = line.decode('unicode_escape')
                line = line.strip()
                line = line.split('::')
                data.append(line)
            self.ratings = pd.DataFrame(data,columns=['UserID','MovieID','Rating','Timestamp'])
            self.ratings['Rating'] = self.ratings['Rating'].astype(int)
            self.ratings['UserID'] = self.ratings['UserID'].astype(int)
            self.ratings['MovieID'] = self.ratings['MovieID'].astype(int)

    def users(self):
        # Extract users data
        with open(self.users_file,'rb') as f:
            data=[]
            for line in f.readlines():
                line = line.decode('unicode_escape')
                line = line.strip()
                line = line.split('::')
                data.append(line)
            self.users = pd.DataFrame(data,columns=['UserID','Gender','Age','Occupation','Zip-code'])
            self.users['UserID'] = self.users['UserID'].astype(int)

class Item_recommendation_system:
    def __init__(self,data):
        self.data = data

    def calculate_similarity_user(self,user_id):
        similarity_arr=pd.DataFrame(index=self.data.unbiased_data_matrix.columns,columns=['Similarity'])
        print(len(self.data.unbiased_data_matrix.columns),len(self.data.unbiased_data_matrix.index))
        for i in self.data.unbiased_data_matrix.columns:
            similarity_arr.loc[i] = self.calculate_similarity(self.data.unbiased_data_matrix[user_id],self.data.unbiased_data_matrix[i])
        # similarity_arr = pd.DataFrame.from_dict(valp,orient='index',columns=['Similarity'])
        similarity_arr.to_csv('similarity_matrix.csv')
        return similarity_arr

    def calculate_similarity(self,user1,user2):
        # Calculate similarity between two users
        num=0
        den1=0
        den2=0
        for i in user1.index:
            if not np.isnan(user1[i]) and not np.isnan(user2[i]):
                num += user1[i]*user2[i]
                den1 += user1[i]**2
                den2 += user2[i]**2
        den = (den1*den2)**0.5
        if den==0:
            return 0
        else:
            return num/den
                

    def get_error(self):
        # Calculate error
        pass

    def get_expected_rating(self):
        # Calculate expected rating
        pass

    def get_top_n_recommendations(self,user_id,n=5):
        # Get top n recommendations
        # similarity = self.calculate_similarity_user(user_id)
        similarity = pd.read_csv('similarity_matrix.csv',index_col=0)
        predicted_rating = {}
        for j in self.data.unbiased_data_matrix.index:
            num=0
            den=0
            for i in self.data.unbiased_data_matrix.columns:
                if user_id!=i:
                    user_rat_norm=self.data.unbiased_data_matrix[i]
                    sim = similarity.loc[i]['Similarity']
                    if not np.isnan(user_rat_norm[j]):
                        num = user_rat_norm[j]*sim
                        den += abs(sim)
            predicted_rating[j] = num/den
        predicted_rating = pd.DataFrame.from_dict(predicted_rating,orient='index',columns=['Predicted_Rating'])
        predicted_rating += self.data.bias.loc[user_id]['Bias']
        predicted_rating.to_csv('predicted_rating.csv')
        top_n = predicted_rating.sort_values('Predicted_Rating',ascending=False)
        top_n = top_n.head(n)
        return top_n


d=Data('./ml-1m/movies.dat','./ml-1m/ratings.dat','./ml-1m/users.dat')
recommender=Item_recommendation_system(d)
print(recommender.get_top_n_recommendations(1))
# data_matrix=d.data_matrix


# print(data_matrix.head())