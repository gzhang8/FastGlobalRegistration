
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/common/distances.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

#include "helper.h"

#include "../FastGlobalRegistration/app.h"

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
using namespace std;

void getFitness(PointCloudT::Ptr src_ptr,
                PointCloudT::Ptr dst_ptr,
                float corr_dist_threshold_,
                const Eigen::Matrix4f& T,
                size_t &inlier_size , float& fitness_score);

std::string dir_name = "";

void do_all( int num, bool create_odom)
{
  //num = 5;
	Configuration config;

	RGBDTrajectory traj;

  std::vector<Points> hdpcds(num);
  std::vector<Feature> fea_pcds(num);
  std::vector<PointCloudT::Ptr> pcl_pcds(num);

  // preload all pcds and 
  for (int i = 0; i < num; i++) {

    // Load object and scene
    pcl::console::print_highlight( "Load fragments %d.\n", i );

    PointCloudT::Ptr scene (new PointCloudT);
    FeatureCloudT::Ptr scene_features (new FeatureCloudT);
    pcl::console::print_highlight ("Loading point clouds...\n");

    char filename[1024];
    sprintf( filename, "%scloud_bin_%d.pcd", dir_name.c_str(), i );
    std::cout << "\n" << filename << std::endl;
    pcl::io::loadPCDFile<PointNT>( filename, *scene );

    const float leaf = config.resample_leaf_;

    // Downsample
    pcl::console::print_highlight ("Downsampling...\n");
    pcl::VoxelGrid<PointNT> grid;

    // 5cm from config, for each box
    grid.setLeafSize (leaf, leaf, leaf);

    grid.setInputCloud (scene);
    grid.filter (*scene);

    if ( config.estimate_normal_ ) {
      PointCloudT::Ptr scene_bak (new PointCloudT);
      pcl::copyPointCloud( *scene, *scene_bak );

      // Estimate normals for scene
      pcl::console::print_highlight ("Estimating scene normals...\n");
      pcl::NormalEstimationOMP<PointNT,PointNT> nest;
      nest.setRadiusSearch( config.normal_radius_ );
      nest.setInputCloud (scene);
      nest.compute (*scene);

      // normal estimate should agree with normal from file within (-90,90)
      for ( int i = 0; i < scene->size(); i++ ) {
        if ( scene->points[ i ].normal_x * scene_bak->points[ i ].normal_x 
             + scene->points[ i ].normal_y * scene_bak->points[ i ].normal_y
             + scene->points[ i ].normal_z * scene_bak->points[ i ].normal_z < 0.0 ) {
          scene->points[ i ].normal_x *= -1;
          scene->points[ i ].normal_y *= -1;
          scene->points[ i ].normal_z *= -1;
        }
      }
    }

    // Estimate features
    pcl::console::print_highlight ("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch ( config.feature_radius_ );
    fest.setInputCloud (scene);
    fest.setInputNormals (scene);
    fest.compute (*scene_features);

    //pcl::console::print_highlight ("Starting alignment...\n");

    // copy pcd
    hdpcds[i].clear();
    fea_pcds[i].clear();
    int nV = scene->size(), nDim = 33;
    for (int v = 0; v < nV; v++) {
      Vector3f pts_v;
      const pcl::PointNormal &pt = scene->points[v];
      float xyz[3] = {pt.x, pt.y, pt.z};
      //fwrite(xyz, sizeof(float), 3, fid);
      memcpy(&pts_v(0), xyz, 3 * sizeof(float));

      VectorXf feat_v(nDim);
      const pcl::FPFHSignature33 &feature = scene_features->points[v];
      //fwrite(feature.histogram, sizeof(float), 33, fid);
      memcpy(&feat_v(0), feature.histogram, nDim * sizeof(float));

      hdpcds[i].push_back(pts_v);
      fea_pcds[i].push_back(feat_v);
    }
    pcl_pcds[i] = scene;
  }





//#pragma omp parallel for num_threads( 8 ) schedule( dynamic )
	for ( int i = 0; i < num; i++ ) {
		for ( int j = i + 1; j < num; j++ ) {
			// Load object and scene
			pcl::console::print_highlight( "Betweeen fragments %d and %d.\n", i, j );
			bool smart_swapped = false;

      auto scene_h = hdpcds[i];
      auto object_h = hdpcds[j];

      auto scene_features = fea_pcds[i];
      auto object_features = fea_pcds[j];


			const float leaf = config.resample_leaf_;
      std::cout << "pcd size1: " << scene_h.size() << "pcd size 2: " << object_h.size()
                << std::endl;

      // Use larger piece as scene
			if ( config.smart_swap_ ) {
				if ( object_h.size() > scene_h.size() ) {
					//auto temp = object_h;
					object_h = hdpcds[i];
					scene_h = hdpcds[j];

          //auto tmp_fea = object_features;
          object_features = fea_pcds[i];
          scene_features = fea_pcds[j];

					smart_swapped = true;
				} else {
					smart_swapped = false;
				}
			}

      CApp app;
      app.LoadFeature(scene_h, scene_features);
      app.LoadFeature(object_h, object_features);
      app.NormalizePoints();
      app.AdvancedMatching();
      app.OptimizePairwise(true, ITERATION_NUMBER);

      Eigen::Matrix4f* transformation_ptr = new Eigen::Matrix4f();
      *transformation_ptr = app.GetTrans();
      float fitness_score = 10000000;
      size_t inlier_size = 0;
      float inlier_fraction = 0;

      if ( smart_swapped ) {
        getFitness(pcl_pcds[i], pcl_pcds[j], config.max_correspondence_distance_,
                   *transformation_ptr, inlier_size, fitness_score);
        inlier_fraction = static_cast<float>(inlier_size) / static_cast<float>(pcl_pcds[i]->size());
      } else {
        getFitness(pcl_pcds[j], pcl_pcds[i], config.max_correspondence_distance_,
                   *transformation_ptr, inlier_size, fitness_score);
        inlier_fraction = static_cast<float>(inlier_size) / static_cast<float>(pcl_pcds[j]->size());
      }

      bool converged = false;
      if (inlier_fraction >= config.inlier_fraction_|| inlier_size > config.inlier_number_){
        converged = true;
      }

			if (converged) {

        if ( smart_swapped ) {
          *transformation_ptr = transformation_ptr->inverse().eval();
					//information = align.information_target_;
        } else {
					//information = align.information_source_;
        }
        Eigen::Matrix4f& transformation = *transformation_ptr;
//#pragma omp critical
        {
          // Print results
          pcl::console::print_info ("\n");
          pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
          pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
          pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
          pcl::console::print_info ("\n");
          pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
          pcl::console::print_info ("\n");
          //pcl::console::print_info ("Inliers: %d/%i\n", align.getInlierCount(), object_h->size ());

          traj.data_.push_back( FramedTransformation(
                                  i, j, num, (*transformation_ptr).cast< double >() ) );
        }
			} else {
				pcl::console::print_error ("Alignment failed!\n");
			}

      delete transformation_ptr;
		}
	}
	traj.SaveToFile( "result.txt" );
}



void getFitness(PointCloudT::Ptr src_ptr,
                PointCloudT::Ptr dst_ptr,
                float corr_dist_threshold_,
                const Eigen::Matrix4f& T,
                size_t &inlier_size , float& fitness_score)
{
  inlier_size = 0;
	// Initialize variables
	//inliers.clear ();
	//inliers.reserve (input_->size ());
	//fitness_score = 0.0f;

	//inliers_target.clear();
	//inliers_target.reserve( target_->size() );

	// Use squared distance for comparison with NN search results
	const float max_range = corr_dist_threshold_ * corr_dist_threshold_;

	// Transform the input dataset using the final transformation
	PointCloudT src_transformed;
	src_transformed.resize (src_ptr->size ());
	transformPointCloud (*src_ptr, src_transformed, T);

  // set KDTree for NN search
  pcl::KdTreeFLANN<PointNT>::Ptr pts_tree (new pcl::KdTreeFLANN<PointNT>);
  PointCloudT::ConstPtr dst_const_ptr(dst_ptr);
  pts_tree->setInputCloud(dst_const_ptr);

	// For each point in the source dataset
	for (size_t i = 0; i < src_transformed.points.size (); ++i)
	{
		// Find its nearest neighbor in the target
		std::vector<int> nn_indices (1);
		std::vector<float> nn_dists (1);
		pts_tree->nearestKSearch (src_transformed.points[i], 1, nn_indices, nn_dists);

		// Check if point is an inlier
		if (nn_dists[0] < max_range)
		{
			// Update inliers
			//inliers.push_back (static_cast<int> (i));
			//inliers_target.push_back( nn_indices[ 0 ] );

			// Update fitness score
      inlier_size ++;
			fitness_score += nn_dists[0];
		}
	}

	// Calculate MSE
	if (inlier_size != 0)
		fitness_score /= static_cast<float> (inlier_size);
	else
		fitness_score = std::numeric_limits<float>::max();
}



int main(int argc, char * argv[])
{

  pcl::ScopeTime t("whole registration");
	if ( argc < 2 ) {
		cout << "Usage : " << endl;
		cout << "    GlobalRegistration.exe <dir>" << endl;
		cout << "    GlobalRegistration.exe <dir> <100-0.log> <segment_length>" << endl;
		return 0;
	}

  int fragment;
  //RGBDTrajectory segment_traj;
  //RGBDTrajectory init_traj;
  //RGBDTrajectory pose_traj;
  //RGBDTrajectory odometry_traj;
  //RGBDInformation odometry_info;

	dir_name = std::string( argv[ 1 ] );
	int num_of_pcds =  std::count_if(
    boost::filesystem::directory_iterator( boost::filesystem::path( dir_name ) ),
		boost::filesystem::directory_iterator(), 
		[](const boost::filesystem::directory_entry& e) {
			return e.path().extension() == ".pcd";  }
	);
	cout << num_of_pcds << " detected." << endl << endl;

	if ( argc == 2 ) {
    //RGBDTrajectory pose_traj;
    //string odom_file_name = dir_name + "/odometry.log";
    //LoadOdomPoses(odom_file_name, pose_traj);
		do_all( num_of_pcds, true );
	}

	return 0;
}
