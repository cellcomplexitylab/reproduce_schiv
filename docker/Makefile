all: schiv_docker_image_available

# Try to pull docker image, build new if fails.
schiv_docker_image_available:
	if docker pull gui11aume/schiv && docker tag gui11aume/schiv schiv; then \
		touch schiv_docker_image_available; \
	else if docker build . -t schiv; then \
		touch schiv_docker_image_available; \
	fi fi

clean:
	rm schiv_docker_image_available && docker rmi -f schiv gui11aume/schiv
