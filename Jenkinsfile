pipeline {
    agent any

    environment {
        // 모델을 Git에 넣지 않고, 배포 시점에 workspace로 가져오는 방식
        // 기본 위치: models/skin_model_final/best.pt  (컨테이너는 -v $(pwd)/models:/app/models 로 마운트)
        MODEL_DIR = "models/skin_model_final"
        MODEL_FILE = "best.pt"
        MODEL_PATH = "${MODEL_DIR}/${MODEL_FILE}"

        // 방법 1) URL 다운로드 (예: S3/MinIO/파일서버). 비어있으면 다운로드 시도 안 함.
        MODEL_DOWNLOAD_URL = "${env.MODEL_DOWNLOAD_URL}"

        // 방법 2) Jenkins Credentials "Secret file" ID로 업로드한 모델 파일을 복사
        // credentialsId 예: 'petmediscan_skin_best_pt'
        MODEL_SECRET_FILE_CREDENTIALS_ID = "petmediscan_skin_best_pt"
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '📦 Git Repository 체크아웃 중...'
                checkout scm
            }
        }

        stage('Prepare Model') {
            steps {
                echo '🧠 모델 파일 준비 중...'
                sh '''
                    set -e
                    mkdir -p "${MODEL_DIR}"

                    if [ -f "${MODEL_PATH}" ]; then
                      echo "✅ 모델 파일 이미 존재: ${MODEL_PATH}"
                      ls -la "${MODEL_PATH}"
                      exit 0
                    fi

                    # (A) Jenkins Secret file credentials로 주입
                    if [ -n "${MODEL_SECRET_FILE_CREDENTIALS_ID}" ]; then
                      echo "🔐 Jenkins Credentials(Secret file)에서 모델 가져오는 중..."
                    fi
                '''

                script {
                    if (env.MODEL_SECRET_FILE_CREDENTIALS_ID?.trim()) {
                        withCredentials([file(credentialsId: env.MODEL_SECRET_FILE_CREDENTIALS_ID, variable: 'MODEL_SECRET_FILE')]) {
                            sh '''
                                set -e
                                mkdir -p "${MODEL_DIR}"
                                cp "$MODEL_SECRET_FILE" "${MODEL_PATH}"
                                echo "✅ Secret file -> ${MODEL_PATH}"
                                ls -la "${MODEL_PATH}"
                            '''
                        }
                    } else {
                        sh '''
                            set -e
                            if [ -f "${MODEL_PATH}" ]; then
                              exit 0
                            fi

                            # (B) URL에서 다운로드
                            if [ -n "${MODEL_DOWNLOAD_URL}" ]; then
                              echo "⬇️  URL에서 모델 다운로드: ${MODEL_DOWNLOAD_URL}"
                              curl -fL "${MODEL_DOWNLOAD_URL}" -o "${MODEL_PATH}"
                              echo "✅ 다운로드 완료: ${MODEL_PATH}"
                              ls -la "${MODEL_PATH}"
                              exit 0
                            fi

                            echo "❌ 모델 파일이 없습니다: ${MODEL_PATH}"
                            echo "   해결 방법:"
                            echo "   - workspace에 직접 ${MODEL_PATH} 배치, 또는"
                            echo "   - 환경변수 MODEL_DOWNLOAD_URL 설정, 또는"
                            echo "   - 환경변수 MODEL_SECRET_FILE_CREDENTIALS_ID 설정(Secret file credentials)"
                            exit 1
                        '''
                    }
                }
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
                        --network petmediscan-infra_petmediscan-network \
                        -p 5001:5001 \
                        -e MODEL_VERSION=skin_model_final \
                        -e MODEL_PATH=/app/models/skin_model_final/best.pt \
                        -e SERVICE_TYPE=skin \
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
