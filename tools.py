
import os
import random
from mltools.anomaly_detection import AnomalyDetectionIsolationForest
from mltools.assoc_rule_mining import AssociationRuleMining
from mltools.clustering import KMeansClustering
from mltools.density_estimation import DensityEstimationKDE
from mltools.dimensionality_reduction import DimensionalityReductionPCA
from mltools.gan import SimpleGAN
from mltools.lda import LDATopicModel
from mltools.nmf import NMFTopicModel
#from mltools.rnn import 
from mltools.self_organizing_maps import SelfOrganizingMap
from mltools.vae import SimpleVAE
from mltools.word_embeddings import WordEmbeddings


class MLFunctionSelector:
    def __init__(self, input_data=""):
        self.ml_functions = ["anomaly_detection", "assoc_rule_mining", "k_means_clustering", 
            "kde", "pca", "gan", "lda", "nmf", "self_organizing_maps", "word_embeddings", "vae"]
        self.input_data = input_data
        self.seed=42,
        self.num_samples=100,
        self.num_samples=0.05,
        self.association_min_support=0.4,
        self.association_min_confidence=0.7,
        self.kmeans_n_clusters=4,
        self.kde_bandwidth=0.5,
        self.pca_n_components=2,
        self.gan_input_dim=2,
        self.gan_generator_output_dim=2,
        self.gan_discriminator_output_dim=1,
        self.gan_epochs=200,
        self.gan_batch_size=32,
        self.lda_documents=None,
        self.nmf_documents=None,
        self.som_data=None,
        self.vae_input_dim=128,
        self.vae_latent_dim=32,
        self.word_embedding_model_name='word2vec-google-news-300'


    def select_function(self, selected_function=""):
        eval_func = f"self.{selected_function}()"
        print(f"eval: {eval_func}")
        result = eval(eval_func)
        return f"Selected ML Function: {selected_function}\nResult: {result}"


    def anomaly_detection(self):
        try:
            # Generate sample data
            data = generate_sample_data(self.seed, self.num_samples)
            
            # Anomaly Detection
            anomaly_detector = AnomalyDetectionIsolationForest(contamination=self.anomaly_contamination, random_state=self.seed)
            anomaly_scores = anomaly_detector.fit_predict(data)
            anomaly_detector.plot_anomalies(data)
            return anomaly_scores
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def assoc_rule_mining(self):
        try:
            # Association Rule Mining
            transactions = [{'apple', 'banana', 'cherry'}, {'banana', 'cherry'}, {'apple', 'banana'},
                            {'apple', 'cherry'}, {'apple', 'banana', 'cherry'}, {'banana'}, {'cherry'}]
            association_miner = AssociationRuleMining(min_support=self.association_min_support, min_confidence=self.association_min_confidence)
            association_miner.fit(transactions)
            association_miner.display_rules()
            return association_miner
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def k_means_clustering(self):
        try:
            # K-Means Clustering
            kmeans_data, _ = make_blobs(n_samples=self.num_samples, centers=self.kmeans_n_clusters, random_state=self.seed)
            kmeans_clusterer = KMeansClustering(n_clusters=self.kmeans_n_clusters, random_state=self.seed)
            kmeans_clusterer.fit(kmeans_data)
            kmeans_clusterer.plot_clusters(kmeans_data)
            return kmeans_clusterer
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def kde(self):
        try:
            # Kernel Density Estimation
            kde_estimator = DensityEstimationKDE(bandwidth=self.kde_bandwidth)
            kde_estimator.fit(data)
            kde_estimator.plot_density()
            return kde_estimator
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def pca(self):
        try:
            # PCA Dimensionality Reduction
            pca_reducer = DimensionalityReductionPCA(n_components=self.pca_n_components)
            reduced_data = pca_reducer.fit_transform(data)
            pca_reducer.plot_variance_explained()
            return pca_reducer
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def gan(self):
        try:
            # GAN (Generative Adversarial Network)
            data_gan = np.random.randn(self.num_samples, self.gan_input_dim)
            gan = SimpleGAN(input_dim=self.gan_input_dim, latent_dim=self.gan_generator_output_dim)
            gan.train(data_gan, epochs=self.gan_epochs, batch_size=self.gan_batch_size)
            generated_data = gan.generate_samples(num_samples=self.num_samples)
            plt.figure(figsize=(6, 6))
            plt.scatter(generated_data[:, 0], generated_data[:, 1])
            plt.title('Generated Data from GAN')
            plt.show()
            return generated_data
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def lda(self):
        try:
            # LDA Topic Modeling
            if self.lda_documents:
                lda_topic_model = LDATopicModel(n_topics=3)
                document_term_matrix, lda_model = lda_topic_model.fit_transform(self.lda_documents)
                feature_names = lda_topic_model.vectorizer.get_feature_names_out()
                lda_topic_model.display_topics(feature_names)
                return document_term_matrix
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def nmf(self):
        try:
            # NMF Topic Modeling
            if self.nmf_documents:
                nmf_topic_model = NMFTopicModel(n_topics=3)
                document_term_matrix, nmf_model = nmf_topic_model.fit_transform(self.nmf_documents)
                feature_names = nmf_topic_model.vectorizer.get_feature_names_out()
                nmf_topic_model.display_topics(feature_names)
                return document_term_matrix
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def self_organizing_maps(self):
        try:
            # Self-Organizing Map
            if self.som_data:
                som = SelfOrganizingMap(grid_size=(5, 5), input_dim=2)
                som.train(self.som_data)
                som.plot(self.som_data)
                return som
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def word_embeddings(self):
        try:
            # Word Embeddings
            if self.word_embedding_model_name:
                embeddings = ""
                word_embeddings = WordEmbeddings(embedding_model_name=self.word_embedding_model_name)
                word_embeddings.load_embedding_model()
                word_vector = word_embeddings.get_embedding('king')
                similarity_score = word_embeddings.get_similarity('king', 'queen')
                embeddings += f'Word Embedding Vector for "king": {word_vector}'
                embeddings += f'Similarity Score between "king" and "queen": {similarity_score}'
                return embeddings
        except Exception as e:
            print(f"Error in toolset: {e}")
    
    def vae(self):
        try:
            # VAE (Variational Autoencoder)
            vae_data = np.random.randn(self.num_samples, self.vae_input_dim)
            vae = SimpleVAE(input_dim=self.vae_input_dim, latent_dim=self.vae_latent_dim)
            vae.train(vae_data, epochs=100, batch_size=32)
            generated_samples = vae.generate_samples(num_samples=self.num_samples)
            plt.figure(figsize=(6, 6))
            plt.scatter(generated_samples[:, 0], generated_samples[:, 1])
            plt.title('Generated Samples from VAE')
            plt.show()    
            return generated_samples
        except Exception as e:
            print(f"Error in toolset: {e}")

