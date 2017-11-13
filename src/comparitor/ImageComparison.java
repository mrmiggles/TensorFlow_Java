package comparitor;

import java.nio.file.Paths;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import classifier.TFModels;
import general.DirectoryMethods;

public class ImageComparison {

	public static void main(String[] args) {
	    //String modelDir = TFModels.getMobilenetDirectory(); 	      
	    //String pbName = TFModels.getMobilenetV1_1_PBName();
	    //String labelFile = TFModels.getMobilenetV1_1_Lables(); 
	    
	    String outputLayer = "MobilenetV1/Predictions/Reshape_1";
	    
	    String modelDir = TFModels.getInception_v3_Directory();
	    String pbName = TFModels.getInception_v3_PBName();
	    
	    String imageDir = "C:\\Users\\" + TFModels.user + "\\Desktop\\Alexie\\";
	    String imageFile = imageDir + "Subject.jpg";
	    byte[] imageBytes = DirectoryMethods.readAllBytesOrExit(Paths.get(imageFile));
	    
	    
		byte[] graphDef = DirectoryMethods.readAllBytesOrExit(Paths.get(modelDir, pbName));
	    try (Graph g = new Graph()) {
		      g.importGraphDef(graphDef);
		      
		      
		      try (Session s = new Session(g)) {
		    	  
		          //Tensor result = s.runner().feed("input", image).fetch(outputLayer).run().get(0)
		    	  Tensor image = constructAndExecutreGraphToGetKeypoints(imageBytes);
		    	  Tensor result = s.runner().feed("DecodeJpeg/contents:0", image).fetch("pool_3:0").run().get(0);
		    	  //System.out.println(result.toString());
		    	  
		      }
	    }
	}
	
	private static Tensor constructAndExecutreGraphToGetKeypoints(byte[] imageBytes) {
	    try (Graph g = new Graph()) {
		      Builder.GraphBuilder b = new Builder.GraphBuilder(g);
		      // Some constants specific to the pre-trained model at:
		      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
		      

		      // Since the graph is being constructed once per execution here, we can use a constant for the
		      // input image. If the graph were to be re-used for multiple input images, a placeholder would
		      // have been more appropriate.
		      final Output input = b.constant("input", imageBytes);
		      final Output output = b.decodeJpeg(input, 3);
		      	System.out.println(output.op().name());
		      try (Session s = new Session(g)) {
		        return s.runner().fetch(output.op().name()).run().get(0);
		      }
		    }		
		
	}

}
