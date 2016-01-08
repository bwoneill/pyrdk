import org.apache.spark.mllib.clustering.dbscan.DBSCAN
import com.amazonaws.services.s3._

object DBSCANSample {

  def main(args: Array[String]) {

//    val conf = new SparkConf().setAppName("RDK DBSCAN")
//    val sc = new SparkContext(conf)

    val data = sc.textFile("fit_rdd.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
    val model = DBSCAN.fit(
      parsedData,
      eps = 0.01,
      minPoints = 5,
      maxPointsPerPartition = 512)

    model.labeledPoints.map(p =>  s"${p.x},${p.y},${p.cluster}").saveAsTextFile("dbscan.txt")

    sc.stop()
  }
}