import pickle
import imageresize
#model = imageresize.loadingmodel()
#pickle.dump( model, open( "save.p", "wb" ) )
#favorite_color = pickle.load( open( "save.p", "rb" ) )


#import pandas as pd
#email = "sample@gmail.com"
#passwrd = "xyz"
#data = [[email, passwrd]]
#df = pd.DataFrame(data, columns = ['Email', 'Password'])
#pickle.dump(df, open( "signupdataframe.p", "wb" ))
df = pickle.load( open( "signupdataframe.p", "rb"))
print(df)