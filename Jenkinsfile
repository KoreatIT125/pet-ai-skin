pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo '📦 Git Repository 체크아웃 중...'
                checkout scm
            }
        }
        
        stage('Docker Build') {
            steps {
                echo '🐳 Docker 이미지 빌드 중...'
                sh '''
                    docker build -t petmediscan-ai-skin:${BUILD_NUMBER} .
                    docker tag petmediscan-ai-skin:${BUILD_NUMBER} petmediscan-ai-skin:latest
                '''
            }
        }
        
        stage('Deploy') {
            steps {
                echo '🚀 Docker 컨테이너 재배포 중...'
                sh '''
                    docker stop petmediscan-ai-skin || true
                    docker rm petmediscan-ai-skin || true
                    
                    docker run -d \
                        --name petmediscan-ai-skin \
                        --network pet-infra_petmediscan-network \
                        -p 5001:5001 \
                        -v $(pwd)/models:/app/models \
                        petmediscan-ai-skin:latest
                '''
            }
        }
    }
    
    post {
        success {
            echo '✅ AI Skin 빌드 및 배포 성공!'
        }
        failure {
            echo '❌ AI Skin 빌드 또는 배포 실패!'
        }
        always {
            echo '🧹 워크스페이스 정리 중...'
            cleanWs()
        }
    }
}
