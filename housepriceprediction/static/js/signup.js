console.log("hi");
const form = document.querySelector("#form");

const usernameInput = document.querySelector("#username");
const userMail = document.querySelector("#email");
const passwordInput = document.querySelector("#password");
const passwordConfirmInput = document.querySelector("#confirm_pass");
form.addEventListener("submit", (e) => {
  // e.preventDefault();
  validateForm();
});

function setError(element, errorMessage) {
  const inputControl = element.parentElement;
  const errorDisplay = inputControl.querySelector(".error");
  errorDisplay.innerText = errorMessage;
  inputControl.classList.add("error");
  inputControl.classList.remove("success");
}
function setSucess(element) {
  const inputControl = element.parentElement;
  const errorDisplay = inputControl.querySelector(".error");
  errorDisplay.innerText = "";
  inputControl.classList.add("success");
  inputControl.classList.remove("error");
}
const validateForm = () => {
  if (usernameInput.value === "") {
    setError(usernameInput, "Username is required");
  } else if (usernameInput.value.length < 6) {
    setError(usernameInput, "username must be more then 5 characters.");
  } else {
    setSucess(usernameInput);
  }

  if (userMail.value === "") {
    setError(userMail, "Email is required");
  } else {
    setSucess(userMail);
  }

  if (passwordInput.value === "") {
    setError(passwordInput, "Password is Required");
  } else if (passwordInput.value.length < 7) {
    setError(passwordInput, "Password must be more then 6 characters");
  } else {
    setSucess(passwordInput);
  }

  if (passwordConfirmInput.value === "") {
    setError(passwordConfirmInput, "Password Conformation is Required");
  } else if (passwordConfirmInput.value !== passwordInput.value) {
    setError(passwordConfirmInput, "Passwords must match");
  } else {
    setSucess(passwordConfirmInput);
  }
};
