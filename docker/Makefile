all: schiv_docker_image_available

# Try to pull docker image, build new if fails.
schiv_docker_image_available:
	if docker pull gui11aume/schiv; then \
		touch schiv_docker_image_available; \
	else if docker build . -t schiv; then \
		touch schiv_docker_image_available; \
	fi fi

clean:
	rm schiv_docker_image_available && docker rmi schiv
