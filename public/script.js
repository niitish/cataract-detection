const formImageElement = document.getElementById("image");
const previewTextElement = document.getElementById("preview_text");
const image = document.getElementById("preview_img");

const clearThings = () => {
  formImageElement.value = "";
  previewTextElement.innerText = "";
  image.src = "";
};

clearThings();

formImageElement.onchange = () => {
  const file = formImageElement.files[0];

  if (file.type !== "image/png" && file.type !== "image/jpeg") {
    clearThings();
    return alert("Only PNG and JPEG images are allowed.");
  }

  previewTextElement.innerText = "Preview";
  const reader = new FileReader();
  reader.onload = (event) => {
    image.src = event.target.result;
  };
  reader.readAsDataURL(file);
};
