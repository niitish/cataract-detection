const ham = document.getElementById("ham");
const closeHam = document.getElementById("close-btn");

ham.addEventListener("click", () => {
	ham.nextElementSibling.classList.add("active");
});

closeHam.addEventListener("click", () => {
	ham.nextElementSibling.classList.remove("active");
});
