// Class that stores all the necessary local mpi info such as coordinates and neighbors

class mpi_decomposition {

public:

    int mpi_rank;
    int mpi_size;
    int mpi_max_rank[2]; // Array to store the maximum ranks in X and Y dimensions
    int mpi_coords[2]; // Coordinates corresponding to the mpi_rank
    int neighbors_mpi_rank[4]; // Array to store ranks of the neighbors: right, left, bottom, top

    mpi_decomposition(int rank, int size, int max_rank[2]) {
        mpi_rank = rank;
        mpi_size = size;
        mpi_max_rank[0] = max_rank[0];
        mpi_max_rank[1] = max_rank[1];

        // Validate the configuration
        if (mpi_max_rank[0] * mpi_max_rank[1] != mpi_size) {
            throw std::invalid_argument("The product of mpi_max_rank dimensions does not match mpi_size.");
        }

        // Compute coordinates and neighbors
        computeCoords();
        computeNeighborRanks();
    }

    void computeCoords() {
        mpi_coords[0] = mpi_rank / mpi_max_rank[1];
        mpi_coords[1] = mpi_rank % mpi_max_rank[1];
    }

    int coordsToRank(const int coords[2]) {
        return coords[0] * mpi_max_rank[1] + coords[1];
    }

    void computeNeighborRanks() {
        // Right neighbor
        int right[2] = {(mpi_coords[0] + 1) % mpi_max_rank[0], mpi_coords[1]};
        neighbors_mpi_rank[0] = coordsToRank(right);

        // Left neighbor
        int left[2] = {(mpi_coords[0] + mpi_max_rank[0] - 1) % mpi_max_rank[0], mpi_coords[1]};
        neighbors_mpi_rank[1] = coordsToRank(left);

        // Bottom neighbor
        int bottom[2] = {mpi_coords[0], (mpi_coords[1] + 1) % mpi_max_rank[1]};
        neighbors_mpi_rank[2] = coordsToRank(bottom);

        // Top neighbor
        int top[2] = {mpi_coords[0], (mpi_coords[1] + mpi_max_rank[1] - 1) % mpi_max_rank[1]};
        neighbors_mpi_rank[3] = coordsToRank(top);
    }

    //Print info to user
    void printDetails() {
        std::cout << "Rank: " << mpi_rank << ", Coordinates: (" << mpi_coords[0] << ", " << mpi_coords[1] << "), ";
        std::cout << "Neighbors - Right: " << neighbors_mpi_rank[0] << ", Left: " << neighbors_mpi_rank[1] << ", Bottom: " << neighbors_mpi_rank[2] << ", Top: " << neighbors_mpi_rank[3] << std::endl;
    }
};
