import pandas as pd 
import pickle
import numpy as np
import json 

class PredictEmprestimo(object):

 def __init__(self):
    self.idade                                = pickle.load(open('../notebooks/parameter/idade_scaler.pkl', 'rb'))
    self.renda                                = pickle.load(open('../notebooks/parameter/renda_scaler.pkl', 'rb'))  
    self.tempo_emprego                        = pickle.load(open('../notebooks/parameter/tempo_emprego_scaler.pkl', 'rb'))
    self.valor_emprestimo                     = pickle.load(open('../notebooks/parameter/valor_emprestimo_scaler.pkl', 'rb'))
    self.taxa_juros_emprestimo                = pickle.load(open('../notebooks/parameter/taxa_juros_emprestimo_scaler.pkl', 'rb'))
    self.relacao_emprestimo_renda             = pickle.load(open('../notebooks/parameter/relacao_emprestimo_renda_scaler.pkl', 'rb'))
    self.historico_credito                    = pickle.load(open('../notebooks/parameter/historico_credito_scaler.pkl', 'rb'))
    self.taxa_juros_ajustada_renda            = pickle.load(open('../notebooks/parameter/taxa_juros_ajustada_renda_scaler.pkl', 'rb'))
    self.proporcao_emprestimo_idade           = pickle.load(open('../notebooks/parameter/proporcao_emprestimo_idade_scaler.pkl', 'rb'))
    self.proporcao_emprestimo_tempo_emprego   = pickle.load(open('../notebooks/parameter/proporcao_emprestimo_tempo_emprego_scaler.pkl', 'rb'))
    self.proporcao_renda_emprestimo_historico = pickle.load(open('../notebooks/parameter/proporcao_renda_emprestimo_historico_scaler.pkl', 'rb'))
    self.proporcao_renda_emprestimo           = pickle.load(open('../notebooks/parameter/proporcao_renda_emprestimo_scaler.pkl', 'rb'))
    self.posse_casa                           = pickle.load(open('../notebooks/parameter/posse_casa_encoder.pkl', 'rb'))
    self.finalidade_emprestimo                = pickle.load(open('../notebooks/parameter/finalidade_emprestimo_encoder.pkl', 'rb'))        
    self.grau_risco_emprestimo                = pickle.load(open('../notebooks/parameter/grau_risco_emprestimo_encoder.pkl', 'rb'))
    self.registro_inadimplencia               = pickle.load(open('../notebooks/parameter/registro_inadimplencia_encoder.pkl', 'rb'))
    self.imputer                              = pickle.load(open('../notebooks/parameter/imputer_knn.pkl', 'rb'))



 def data_cleaning(self, df1):
    num_attributes = df1.select_dtypes(include=['int64', 'float64'])
    cat_attributes = df1.select_dtypes(include=['object', 'category'])
    
    if num_attributes.isnull().any().any():
        df_imputed = pd.DataFrame(self.imputer.transform(num_attributes), columns=num_attributes.columns)
        df1 = pd.concat([df_imputed, cat_attributes], axis=1)
    else:
        return df1
    return df1

 def  feature_engineering(self, df2):
    # Taxas de juros ajustadas a renda
    df2['taxa_juros_ajustada_renda'] = df2['taxa_juros_emprestimo'] / df2['renda']


    # Proporção do valor do emprestimo em relacao a idade
    df2['proporcao_emprestimo_idade'] = df2['valor_emprestimo'] / df2['idade']

    # Proporção do valor do emprestimo em relacao ao tempo de emprego
    df2['tempo_emprego'] = np.where(df2['tempo_emprego'] == 0, 0.5, df2['tempo_emprego'])  # Evitar divisão por zero
    df2['proporcao_emprestimo_tempo_emprego'] = df2['valor_emprestimo'] / df2['tempo_emprego']

    # Proporção renda valor emprestimo
    df2['proporcao_renda_emprestimo'] = df2['renda'] / df2['valor_emprestimo']

    # Categoria renda 
    faixas_renda = [0, 3000, 7000, 50000,float('inf')]
    categorias_renda = ['Baixa', 'Média baixa', 'Media Alta', ' Alta']
    df2['categoria_renda'] = pd.cut(df2['renda'], bins=faixas_renda, labels=categorias_renda) 

    # Categoria idade
    faixas_idade = [0, 30, 50, float('inf')]
    categoria_idade = ['Jovem', 'Meia-idade', 'Idoso']
    df2['categoria_idade'] = pd.cut(df2['idade'], bins=faixas_idade, labels=categoria_idade)

    # Proporção renda valor emprestimo em relacao a historia de credito
    df2['proporcao_renda_emprestimo_historico'] = df2['proporcao_renda_emprestimo'] / df2['historico_credito']
    return df2

 def data_preparation(self, df3):

    df3['idade'] = self.idade.transform(df3[['idade']])
    df3['renda'] = self.renda.transform(df3[['renda']])
    df3['tempo_emprego'] = self.tempo_emprego.transform(df3[['tempo_emprego']])
    df3['valor_emprestimo'] = self.valor_emprestimo.transform(df3[['valor_emprestimo']])
    df3['taxa_juros_emprestimo'] = self.taxa_juros_emprestimo.transform(df3[['taxa_juros_emprestimo']])
    df3['relacao_emprestimo_renda'] = self.relacao_emprestimo_renda.transform(df3[['relacao_emprestimo_renda']])
    df3['historico_credito'] = self.historico_credito.transform(df3[['historico_credito']])
    df3['taxa_juros_ajustada_renda'] = self.taxa_juros_ajustada_renda.transform(df3[['taxa_juros_ajustada_renda']])
    df3['proporcao_emprestimo_idade'] = self.proporcao_emprestimo_idade.transform(df3[['proporcao_emprestimo_idade']])
    df3['proporcao_emprestimo_tempo_emprego'] = self.proporcao_emprestimo_tempo_emprego.transform(df3[['proporcao_emprestimo_tempo_emprego']])    
    df3['proporcao_renda_emprestimo_historico'] = self.proporcao_renda_emprestimo_historico.transform(df3[['proporcao_renda_emprestimo_historico']])
    df3['proporcao_renda_emprestimo'] = self.proporcao_renda_emprestimo.transform(df3[['proporcao_renda_emprestimo']])
    df3['registro_inadimplencia'] = self.registro_inadimplencia.transform(df3['registro_inadimplencia'])

    
   
    df3['posse_casa'] = self.posse_casa.transform(df3['posse_casa'])
    df3['finalidade_emprestimo'] = self.finalidade_emprestimo.transform(df3['finalidade_emprestimo'])
    df3['grau_risco_emprestimo'] = self.grau_risco_emprestimo.transform(df3['grau_risco_emprestimo'])



    # categoria renda 
    renda_dict = {
        'Baixa': 1,
        'Média baixa': 2,
        'Media Alta': 3,  # Corrigido: com acento
        ' Alta': 4
    }

    df3['categoria_renda'] = df3['categoria_renda'].map(renda_dict)
    # Convertendo a coluna 'categoria_renda' para o tipo inteiro
    df3['categoria_renda'] = df3['categoria_renda'].astype(int)  # Adicionado fillna para segurança


    # categoria idade
    idade_dict = {
        'Jovem': 1,
        'Meia-idade': 2,
        'Idoso': 3
    }   
   






    df3['categoria_idade'] = df3['categoria_idade'].map(idade_dict)
    # Convertendo a coluna 'categoria_idade' para o tipo inteiro    
    df3['categoria_idade'] = df3['categoria_idade'].astype(int) 


    boruta_columns= ['renda',
                    'tempo_emprego',
                    'valor_emprestimo',
                    'taxa_juros_emprestimo',
                    'relacao_emprestimo_renda',
                    'posse_casa',
                    'finalidade_emprestimo',
                    'grau_risco_emprestimo',
                    'taxa_juros_ajustada_renda',
                    'proporcao_emprestimo_idade',
                    'proporcao_emprestimo_tempo_emprego',
                    'proporcao_renda_emprestimo',
                    'proporcao_renda_emprestimo_historico']
    return df3[boruta_columns]



 def get_predictions(self,model,test_data,original_data):
     pred = model.predict(test_data)
     original_data['prediction'] = pred
     
     return original_data.to_json(orient='records')



