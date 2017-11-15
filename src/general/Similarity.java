package general;

public class Similarity {
	
	public static double cosineSimilarity(float[] vectorA, float[] vectorB) {
	    double dotProduct = 0.0;
	    double normA = 0.0;
	    double normB = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	        normA += Math.pow(vectorA[i], 2);
	        normB += Math.pow(vectorB[i], 2);
	    }   

	    return (1 - (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))));
	}
	
	public static double euclideanDistance(float[] sequence1, float[] sequence2) {
	     double sum = 0.0;
	     
	     for (int index=0; index<sequence1.length; index++) {
	          sum = sum + Math.pow((sequence1[index] - sequence2[index]), 2 );
	     }
	     
	     
	     return Math.pow(sum, 0.5);//sum^0.5;
	}	

}
