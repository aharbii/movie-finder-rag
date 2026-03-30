// =============================================================================
// movie-finder-rag — Jenkins declarative pipeline
//
// Triggers:
//   • PR validation  — lint + typecheck + tests through the Docker Makefile
//   • Branch build   — build ingestion image on main
//   • Manual ingest  — trigger production-like ingestion run
//
// Required Jenkins Credential IDs (Canonical setup in infrastructure#9):
//   • qdrant-url           (Secret Text)
//   • qdrant-api-key-rw    (Secret Text)
//   • openai-api-key       (Secret Text)
//   • kaggle-api-token     (Secret Text)
// =============================================================================

pipeline {
    agent any

    environment {
        SERVICE_NAME = "movie-finder-rag"
        DOCKER_BUILDKIT = "1"
    }

    stages {
        stage('Lint + Typecheck') {
            steps {
                script {
                    echo "Starting code quality checks..."
                    sh 'make lint'
                    sh 'make typecheck'
                }
            }
        }

        stage('Test') {
            steps {
                script {
                    echo "Starting unit tests and coverage..."
                    sh 'make test-coverage'
                }
            }
            post {
                always {
                    junit 'test-results.xml'
                    cobertura coberturaReportFile: 'coverage.xml'
                }
            }
        }

        stage('Build Image') {
            when {
                anyOf {
                    branch 'main'
                    buildingTag()
                }
            }
            steps {
                script {
                    echo "Building production-like runtime image..."
                    sh "docker build --target runtime -t ${SERVICE_NAME}:latest ."
                }
            }
        }

        stage('Ingest') {
            when {
                expression { params.RUN_INGESTION == true }
            }
            environment {
                QDRANT_URL = credentials('qdrant-url')
                QDRANT_API_KEY_RW = credentials('qdrant-api-key-rw')
                QDRANT_COLLECTION_NAME = params.COLLECTION_NAME ?: 'movies'
                OPENAI_API_KEY = credentials('openai-api-key')
                KAGGLE_API_TOKEN = credentials('kaggle-api-token')
            }
            steps {
                script {
                    echo "Running offline ingestion pipeline against Qdrant Cloud..."
                    sh 'make ingest'
                }
            }
        }
    }

    post {
        always {
            sh 'make ci-down || true'
            cleanWs()
        }
        failure {
            echo "Pipeline failed on branch ${env.BRANCH_NAME} — check logs above."
        }
    }
}
