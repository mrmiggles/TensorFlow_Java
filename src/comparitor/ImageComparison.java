package comparitor;

import java.nio.file.Paths;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import classifier.TFModels;
import general.DirectoryMethods;

public class ImageComparison {

	public static void main(String[] args) {
	    String modelDir = TFModels.getMobilenetDirectory(); 
	    //String pbName = TFModels.getMobilenetV1_050_PBName(); 
	    //String labelFile = TFModels.getMobilenetV1_050_Lables(); 
	      
	    String pbName = TFModels.getMobilenetV1_1_PBName();
	    String labelFile = TFModels.getMobilenetV1_1_Lables(); 
	    
	    String outputLayer = "MobilenetV1/Predictions/Reshape_1";
	    
		byte[] graphDef = DirectoryMethods.readAllBytesOrExit(Paths.get(modelDir, pbName));
	    try (Graph g = new Graph()) {
		      g.importGraphDef(graphDef);
		      try (Session s = new Session(g);
		          //Tensor result = s.runner().feed("input", image).fetch(outputLayer).run().get(0)
		    		  ) {
		      }
	    }

	}

}
