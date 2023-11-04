import os
import random

# Import all ML function classes from the 'functions' directory
from functions import *

class MLFunctionSelector:
    def __init__(self):
        pass
        # Create instances of all ML function classes

    def select_function(self, data_input):
        # Use some logic to determine the most appropriate ML function based on data_input
        # For example, you can randomly select a function for demonstration purposes
        selected_function = random.choice(list(self.ml_functions.keys()))

        # If 'toolset' function is selected, call it with the appropriate arguments
        if selected_function == 'Toolset':
            result = toolset(
                seed=42,
                num_samples=100,
                anomaly_contamination=0.05,
                association_min_support=0.4,
                association_min_confidence=0.7,
                kmeans_n_clusters=4,
                kde_bandwidth=0.5,
                pca_n_components=2,
                gan_input_dim=2,
                gan_generator_output_dim=2,
                gan_discriminator_output_dim=1,
                gan_epochs=200,
                gan_batch_size=32,
                lda_documents=None,
                nmf_documents=None,
                som_data=None,
                vae_input_dim=128,
                vae_latent_dim=32,
                word_embedding_model_name='word2vec-google-news-300'
            )
        else:
            # Call the selected ML function with the data_input
            result = self.ml_functions[selected_function].apply(data_input)

        return f"Selected ML Function: {selected_function}\nResult: {result}"

    def toolset(self,
        seed=42,
        num_samples=100,
        anomaly_contamination=0.05,
        association_min_support=0.4,
        association_min_confidence=0.7,
        kmeans_n_clusters=4,
        kde_bandwidth=0.5,
        pca_n_components=2,
        gan_input_dim=2,
        gan_generator_output_dim=2,
        gan_discriminator_output_dim=1,
        gan_epochs=200,
        gan_batch_size=32,
        lda_documents=None,
        nmf_documents=None,
        som_data=None,
        vae_input_dim=128,
        vae_latent_dim=32,
        word_embedding_model_name='word2vec-google-news-300'
    ):
        try:
            # Generate sample data
            data = generate_sample_data(seed, num_samples)
            
            # Anomaly Detection
            anomaly_detector = AnomalyDetectionIsolationForest(contamination=anomaly_contamination, random_state=seed)
            anomaly_scores = anomaly_detector.fit_predict(data)
            anomaly_detector.plot_anomalies(data)
            
            # Association Rule Mining
            transactions = [{'apple', 'banana', 'cherry'}, {'banana', 'cherry'}, {'apple', 'banana'},
                            {'apple', 'cherry'}, {'apple', 'banana', 'cherry'}, {'banana'}, {'cherry'}]
            association_miner = AssociationRuleMining(min_support=association_min_support, min_confidence=association_min_confidence)
            association_miner.fit(transactions)
            association_miner.display_rules()
            
            # K-Means Clustering
            kmeans_data, _ = make_blobs(n_samples=num_samples, centers=kmeans_n_clusters, random_state=seed)
            kmeans_clusterer = KMeansClustering(n_clusters=kmeans_n_clusters, random_state=seed)
            kmeans_clusterer.fit(kmeans_data)
            kmeans_clusterer.plot_clusters(kmeans_data)
            
            # Kernel Density Estimation
            kde_estimator = DensityEstimationKDE(bandwidth=kde_bandwidth)
            kde_estimator.fit(data)
            kde_estimator.plot_density()
            
            # PCA Dimensionality Reduction
            pca_reducer = DimensionalityReductionPCA(n_components=pca_n_components)
            reduced_data = pca_reducer.fit_transform(data)
            pca_reducer.plot_variance_explained()
            
            # GAN (Generative Adversarial Network)
            data_gan = np.random.randn(num_samples, gan_input_dim)
            gan = SimpleGAN(input_dim=gan_input_dim, latent_dim=gan_generator_output_dim)
            gan.train(data_gan, epochs=gan_epochs, batch_size=gan_batch_size)
            generated_data = gan.generate_samples(num_samples=num_samples)
            plt.figure(figsize=(6, 6))
            plt.scatter(generated_data[:, 0], generated_data[:, 1])
            plt.title('Generated Data from GAN')
            plt.show()
            
            # LDA Topic Modeling
            if lda_documents:
                lda_topic_model = LDATopicModel(n_topics=3)
                document_term_matrix, lda_model = lda_topic_model.fit_transform(lda_documents)
                feature_names = lda_topic_model.vectorizer.get_feature_names_out()
                lda_topic_model.display_topics(feature_names)
            
            # NMF Topic Modeling
            if nmf_documents:
                nmf_topic_model = NMFTopicModel(n_topics=3)
                document_term_matrix, nmf_model = nmf_topic_model.fit_transform(nmf_documents)
                feature_names = nmf_topic_model.vectorizer.get_feature_names_out()
                nmf_topic_model.display_topics(feature_names)
            
            # Self-Organizing Map
            if som_data:
                som = SelfOrganizingMap(grid_size=(5, 5), input_dim=2)
                som.train(som_data)
                som.plot(som_data)
            
            # Word Embeddings
            if word_embedding_model_name:
                word_embeddings = WordEmbeddings(embedding_model_name=word_embedding_model_name)
                word_embeddings.load_embedding_model()
                word_vector = word_embeddings.get_embedding('king')
                similarity_score = word_embeddings.get_similarity('king', 'queen')
                print(f'Word Embedding Vector for "king": {word_vector}')
                print(f'Similarity Score between "king" and "queen": {similarity_score}')


            # VAE (Variational Autoencoder)
            vae_data = np.random.randn(num_samples, vae_input_dim)
            vae = SimpleVAE(input_dim=vae_input_dim, latent_dim=vae_latent_dim)
            vae.train(vae_data, epochs=100, batch_size=32)
            generated_samples = vae.generate_samples(num_samples=num_samples)
            plt.figure(figsize=(6, 6))
            plt.scatter(generated_samples[:, 0], generated_samples[:, 1])
            plt.title('Generated Samples from VAE')
            plt.show()    
        
        except Exception as e:
            print(f"Error in toolset: {e}")




if __name__ == "__main__":
    # Create an instance of the MLFunctionSelector
    ml_function_selector = MLFunctionSelector()

    # Example: Random data input (you can replace this with actual data)
    random_data_input = "Your random data input here"

    # Select and run the most appropriate ML function
    result = ml_function_selector.select_function(random_data_input)

    # Print the result
    print(result)